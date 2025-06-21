import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as opensearchserverless from 'aws-cdk-lib/aws-opensearchserverless';
import * as cloudFront from 'aws-cdk-lib/aws-cloudfront';
import * as origins from 'aws-cdk-lib/aws-cloudfront-origins';
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as opensearch from 'aws-cdk-lib/aws-opensearchservice';
import * as path from "path";
import * as sqs from 'aws-cdk-lib/aws-sqs';
import { SqsEventSource } from 'aws-cdk-lib/aws-lambda-event-sources';
import * as lambdaEventSources from 'aws-cdk-lib/aws-lambda-event-sources';
import * as ec2 from 'aws-cdk-lib/aws-ec2';

const projectName = `advanced-rag`; 
const region = process.env.CDK_DEFAULT_REGION;    
const accountId = process.env.CDK_DEFAULT_ACCOUNT;
const bucketName = `storage-for-${projectName}-${accountId}-${region}`; 
const vectorIndexName = projectName

const s3_prefix = 'docs';
const s3_capture_prefix = 'captures';
let opensearch_url = "";

const titan_embedding_v2 = [  // dimension = 1024
  {
    "bedrock_region": "us-west-2", // Oregon
    "model_type": "titan",
    "model_id": "amazon.titan-embed-text-v2:0"
  },
  {
    "bedrock_region": "us-east-1", // N.Virginia
    "model_type": "titan",
    "model_id": "amazon.titan-embed-text-v2:0"
  },
  {
    "bedrock_region": "us-east-2", // Ohio
    "model_type": "titan",
    "model_id": "amazon.titan-embed-text-v2:0"
  }
];
const LLM_embedding = titan_embedding_v2;  //  titan_embedding_v2_single

const max_object_size = 102400000; // 100 MB max size of an object, 50MB(default)
const enableHybridSearch = 'Enable';
const supportedFormat = JSON.stringify(["pdf", "txt", "csv", "pptx", "ppt", "docx", "doc", "xlsx", "py", "js", "md", "jpeg", "jpg", "png"]);  
const enableParentDocumentRetrival = 'Enable';

export class CdkAdvancedRagStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // s3 
    const s3Bucket = new s3.Bucket(this, `storage-${projectName}`,{
      bucketName: bucketName,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
      publicReadAccess: false,
      versioned: false,
      cors: [
        {
          allowedHeaders: ['*'],
          allowedMethods: [
            s3.HttpMethods.GET,
            s3.HttpMethods.POST,
            s3.HttpMethods.PUT,
          ],
          allowedOrigins: ['*'],
        },
      ],
    });
    new cdk.CfnOutput(this, 'bucketName', {
      value: s3Bucket.bucketName,
      description: 'The nmae of bucket',
    });

    // cloudfront for sharing s3
    const distribution_sharing = new cloudFront.Distribution(this, `sharing-for-${projectName}`, {
      defaultBehavior: {
        origin: origins.S3BucketOrigin.withOriginAccessControl(s3Bucket),
        allowedMethods: cloudFront.AllowedMethods.ALLOW_ALL,
        cachePolicy: cloudFront.CachePolicy.CACHING_DISABLED,
        viewerProtocolPolicy: cloudFront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
      },
      priceClass: cloudFront.PriceClass.PRICE_CLASS_200,  
    });
    new cdk.CfnOutput(this, `distribution-sharing-DomainName-for-${projectName}`, {
      value: 'https://'+distribution_sharing.domainName,
      description: 'The domain name of the Distribution Sharing',
    });   

    // Knowledge Base Role
    const roleKnowledgeBase = new iam.Role(this,  `role-knowledge-base-for-${projectName}`, {
      roleName: `role-knowledge-base-for-${projectName}-${region}`,
      assumedBy: new iam.CompositePrincipal(
        new iam.ServicePrincipal("bedrock.amazonaws.com")
      )
    });
    
    const bedrockInvokePolicy = new iam.PolicyStatement({ 
      effect: iam.Effect.ALLOW,
      resources: [`*`],
      actions: ["bedrock:*"],
    });        
    roleKnowledgeBase.attachInlinePolicy( 
      new iam.Policy(this, `bedrock-invoke-policy-for-${projectName}`, {
        statements: [bedrockInvokePolicy],
      }),
    );  
    
    const S3Policy = new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      resources: ['*'],
      actions: ["s3:*"],
    });
    roleKnowledgeBase.attachInlinePolicy( 
      new iam.Policy(this, `s3-policy-knowledge-base-for-${projectName}`, {
        statements: [S3Policy],
      }),
    );      
    const knowledgeBaseOpenSearchPolicy = new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      resources: ['*'],
      actions: ["aoss:APIAccessAll"],
    });
    roleKnowledgeBase.attachInlinePolicy( 
      new iam.Policy(this, `aoss-knowledge-base-policy-for-${projectName}`, {
        statements: [knowledgeBaseOpenSearchPolicy],
      }),
    );  

    // lambda Knowledge Base
    const roleLambdaKnowledgeBase = new iam.Role(this, `role-lambda-knowledge-base-for-${projectName}`, {
      roleName: `role-lambda-knowledge-base-for-${projectName}-${region}`,
      assumedBy: new iam.CompositePrincipal(
        new iam.ServicePrincipal("lambda.amazonaws.com"),
        new iam.ServicePrincipal("bedrock.amazonaws.com"),
      ),
    });
    const CreateLogPolicy = new iam.PolicyStatement({  
      resources: [`arn:aws:logs:${region}:${accountId}:*`],
      actions: ['logs:CreateLogGroup'],
    });        
    roleLambdaKnowledgeBase.attachInlinePolicy( 
      new iam.Policy(this, `log-policy-lambda-knowledge-base-for-${projectName}`, {
        statements: [CreateLogPolicy],
      }),
    );
    const CreateLogStreamPolicy = new iam.PolicyStatement({  
      resources: [`arn:aws:logs:${region}:${accountId}:log-group:/aws/lambda/*`],
      actions: ["logs:CreateLogStream","logs:PutLogEvents"],
    });        
    roleLambdaKnowledgeBase.attachInlinePolicy( 
      new iam.Policy(this, `stream-log-policy-lambda-knowledge-base-for-${projectName}`, {
        statements: [CreateLogStreamPolicy],
      }),
    );      

    // bedrock
    roleLambdaKnowledgeBase.attachInlinePolicy( 
      new iam.Policy(this, `bedrock-invoke-policy-knowledge-base-for-${projectName}`, {
        statements: [bedrockInvokePolicy],
      }),
    );  
    roleLambdaKnowledgeBase.attachInlinePolicy( 
      new iam.Policy(this, `bedrock-agent-opensearch-policy-for-${projectName}`, {
        statements: [knowledgeBaseOpenSearchPolicy],
      }),
    );

    const knowledgeBaseBedrockPolicy = new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      resources: ['*'],
      actions: ["bedrock:*"],
    });
    roleKnowledgeBase.attachInlinePolicy( 
      new iam.Policy(this, `bedrock-policy-knowledge-base-for-${projectName}`, {
        statements: [knowledgeBaseBedrockPolicy],
      }),
    );  
    roleLambdaKnowledgeBase.attachInlinePolicy( 
      new iam.Policy(this, `bedrock-policy-lambda-knowledge-base-for-${projectName}`, {
        statements: [knowledgeBaseBedrockPolicy],
      }),
    );  
    roleLambdaKnowledgeBase.attachInlinePolicy( 
      new iam.Policy(this, `s3-policy-lambda-knowledge-base-for-${projectName}`, {
        statements: [S3Policy],
      }),
    );

    // OpenSearch Serverless
    const collectionName = vectorIndexName
    const OpenSearchCollection = new opensearchserverless.CfnCollection(this, `opensearch-correction-for-${projectName}`, {
      name: collectionName,    
      description: `opensearch correction for ${projectName}`,
      standbyReplicas: 'DISABLED',
      type: 'VECTORSEARCH',
    });
    const collectionArn = OpenSearchCollection.attrArn

    new cdk.CfnOutput(this, `OpensearchCollectionEndpoint-${projectName}`, {
      value: OpenSearchCollection.attrCollectionEndpoint,
      description: 'The endpoint of opensearch correction',
    });

    const encPolicyName = `encription-${projectName}`
    const encPolicy = new opensearchserverless.CfnSecurityPolicy(this, `opensearch-encription-policy-for-${projectName}`, {
      name: encPolicyName,
      type: "encryption",
      description: `opensearch encryption policy for ${projectName}`,
      policy:
        `{"Rules":[{"ResourceType":"collection","Resource":["collection/${collectionName}"]}],"AWSOwnedKey":true}`,
    });
    OpenSearchCollection.addDependency(encPolicy);

    const netPolicyName = `network-${projectName}-${region}`
    const netPolicy = new opensearchserverless.CfnSecurityPolicy(this, `opensearch-network-policy-for-${projectName}`, {
      name: netPolicyName,
      type: 'network',    
      description: `opensearch network policy for ${projectName}`,
      policy: JSON.stringify([
        {
          Rules: [
            {
              ResourceType: "dashboard",
              Resource: [`collection/${collectionName}`],
            },
            {
              ResourceType: "collection",
              Resource: [`collection/${collectionName}`],              
            }
          ],
          AllowFromPublic: true,          
        },
      ]), 
    });
    OpenSearchCollection.addDependency(netPolicy);

    const account = new iam.AccountPrincipal(this.account)
    const dataAccessPolicyName = `data-${projectName}`
    const dataAccessPolicy = new opensearchserverless.CfnAccessPolicy(this, `opensearch-data-collection-policy-for-${projectName}`, {
      name: dataAccessPolicyName,
      type: "data",
      policy: JSON.stringify([
        {
          Rules: [
            {
              Resource: [`collection/${collectionName}`],
              Permission: [
                "aoss:CreateCollectionItems",
                "aoss:DeleteCollectionItems",
                "aoss:UpdateCollectionItems",
                "aoss:DescribeCollectionItems",
              ],
              ResourceType: "collection",
            },
            {
              Resource: [`index/${collectionName}/*`],
              Permission: [
                "aoss:CreateIndex",
                "aoss:DeleteIndex",
                "aoss:UpdateIndex",
                "aoss:DescribeIndex",
                "aoss:ReadDocument",
                "aoss:WriteDocument",
              ], 
              ResourceType: "index",
            }
          ],
          Principal: [
            account.arn,
            roleLambdaKnowledgeBase.roleArn,
            roleKnowledgeBase.roleArn
          ], 
        },
      ]),
    });
    OpenSearchCollection.addDependency(dataAccessPolicy);

    // Managed OpenSearch
    // Permission for OpenSearch
    const domainName = projectName
    const resourceArn = `arn:aws:es:${region}:${accountId}:domain/${domainName}/*`
    
    // Use CfnDomain (L1 construct) for more granular control over OpenSearch configuration
    const domain = new opensearch.CfnDomain(this, 'Domain', {
      engineVersion: 'OpenSearch_2.13',
      domainName: domainName,
      clusterConfig: {
        instanceType: 'r6g.large.search',
        instanceCount: 1,
        dedicatedMasterEnabled: false,
        zoneAwarenessEnabled: false,
      },
      ebsOptions: {
        ebsEnabled: true,
        volumeType: 'gp3',
        volumeSize: 100,
      },
      nodeToNodeEncryptionOptions: {
        enabled: true,
      },
      encryptionAtRestOptions: {
        enabled: true,
      },
      domainEndpointOptions: {
        enforceHttps: true,
      },
      accessPolicies: {
        Version: '2012-10-17',
        Statement: [
          {
            Effect: 'Allow',
            Principal: {
              AWS: `arn:aws:iam::${this.account}:root`,
            },
            Action: 'es:*',
            Resource: `arn:aws:es:${this.region}:${this.account}:domain/${domainName}/*`,
          },
        ],
      },
    });
    new cdk.CfnOutput(this, `Domain-of-OpenSearch-for-${projectName}`, {
      value: domain.attrArn,
      description: 'The arn of OpenSearch Domain',
    });
    new cdk.CfnOutput(this, `Endpoint-of-OpenSearch-for-${projectName}`, {
      value: 'https://'+domain.attrDomainEndpoint,
      description: 'The endpoint of OpenSearch Domain',
    });
    opensearch_url = 'https://'+domain.attrDomainEndpoint;

    // S3 - Lambda(S3 event) - SQS(fifo) - Lambda(document)
    // DLQ
    let dlq:any[] = [];
    for(let i=0;i<LLM_embedding.length;i++) {
      dlq[i] = new sqs.Queue(this, 'DlqS3EventFifo'+i, {
        visibilityTimeout: cdk.Duration.seconds(900),
        queueName: `dlq-s3-event-for-${projectName}-${i}.fifo`,  
        fifo: true,
        contentBasedDeduplication: false,
        deliveryDelay: cdk.Duration.millis(0),
        retentionPeriod: cdk.Duration.days(14)
      });
    }

    // SQS for S3 event (fifo) 
    let queueUrl:string[] = [];
    let queue:any[] = [];
    for(let i=0;i<LLM_embedding.length;i++) {
      queue[i] = new sqs.Queue(this, 'QueueS3EventFifo'+i, {
        visibilityTimeout: cdk.Duration.seconds(900),
        queueName: `queue-s3-event-for-${projectName}-${i}.fifo`,  
        fifo: true,
        contentBasedDeduplication: false,
        deliveryDelay: cdk.Duration.millis(0),
        retentionPeriod: cdk.Duration.days(2),
        deadLetterQueue: {
          maxReceiveCount: 1,
          queue: dlq[i]
        }
      });
      queueUrl.push(queue[i].queueUrl);
    }
    
    // Lambda for s3 event manager
    const lambdaS3eventManager = new lambda.Function(this, `lambda-s3-event-manager-for-${projectName}`, {
      description: 'lambda for s3 event manager',
      functionName: `lambda-s3-event-manager-for-${projectName}`,
      handler: 'lambda_function.lambda_handler',
      runtime: lambda.Runtime.PYTHON_3_11,
      code: lambda.Code.fromAsset(path.join(__dirname, '../../lambda-s3-event-manager')),
      timeout: cdk.Duration.seconds(60),      
      environment: {
        sqsFifoUrl: JSON.stringify(queueUrl),
        nqueue: String(LLM_embedding.length)
      }
    });
    for(let i=0;i<LLM_embedding.length;i++) {
      queue[i].grantSendMessages(lambdaS3eventManager); // permision for SQS putItem
    }

    // s3 event source for docs
    const s3PutEventSourceDocs = new lambdaEventSources.S3EventSource(s3Bucket, {
      events: [
        s3.EventType.OBJECT_CREATED_PUT,
        s3.EventType.OBJECT_REMOVED_DELETE,
        s3.EventType.OBJECT_CREATED_COMPLETE_MULTIPART_UPLOAD
      ],
      filters: [
        { prefix: s3_prefix+'/' }
      ]
    });
    lambdaS3eventManager.addEventSource(s3PutEventSourceDocs);

    // s3 event source for captures
    const s3PutEventSourceCaptures = new lambdaEventSources.S3EventSource(s3Bucket, {
      events: [
        s3.EventType.OBJECT_CREATED_PUT,
        s3.EventType.OBJECT_REMOVED_DELETE,
        s3.EventType.OBJECT_CREATED_COMPLETE_MULTIPART_UPLOAD
      ],
      filters: [
        { prefix: s3_capture_prefix+'/' } 
      ]
    });
    lambdaS3eventManager.addEventSource(s3PutEventSourceCaptures); 

    // Lambda 
    const roleLambdaDocument = new iam.Role(this, `role-lambda-document-for-${projectName}`, {
      roleName: `role-lambda-document-for-${projectName}-${region}`,
      assumedBy: new iam.CompositePrincipal(
        new iam.ServicePrincipal("lambda.amazonaws.com"),
        new iam.ServicePrincipal("bedrock.amazonaws.com"),
      )
    });
    roleLambdaDocument.addManagedPolicy({
      managedPolicyArn: 'arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole',
    });
    const BedrockPolicy = new iam.PolicyStatement({  // policy statement for sagemaker
      resources: ['*'],
      actions: ['bedrock:*'],
    });        
    roleLambdaDocument.attachInlinePolicy( // add bedrock policy
      new iam.Policy(this, `bedrock-policy-lambda-document-for-${projectName}`, {
        statements: [BedrockPolicy],
      }),
    );        
    const lambdaInvokePolicy = new iam.PolicyStatement({ 
      resources: ['*'],
      actions: [
        "lambda:InvokeFunction"
      ],
    });        
    roleLambdaDocument.attachInlinePolicy( 
      new iam.Policy(this, `lambda-invoke-policy-for-${projectName}`, {
        statements: [lambdaInvokePolicy],
      }),
    );

    // OpenSearch Policy for Lambda
    const openSearchPolicy = new iam.PolicyStatement({
      effect: iam.Effect.ALLOW,
      resources: [resourceArn],
      actions: [
        "es:ESHttpGet",
        "es:ESHttpPut", 
        "es:ESHttpPost",
        "es:ESHttpDelete",
        "es:ESHttpHead"
      ],
    });
    roleLambdaDocument.attachInlinePolicy(
      new iam.Policy(this, `opensearch-policy-lambda-document-for-${projectName}`, {
        statements: [openSearchPolicy],
      }),
    );

    // Lambda for document manager
    let lambdDocumentManager:any[] = [];
    for(let i=0;i<LLM_embedding.length;i++) {
      lambdDocumentManager[i] = new lambda.DockerImageFunction(this, `lambda-document-manager-for-${projectName}-${i}`, {
        description: 'S3 document manager',
        functionName: `lambda-document-manager-for-${projectName}-${i}`,
        role: roleLambdaDocument,
        code: lambda.DockerImageCode.fromImageAsset(path.join(__dirname, '../../lambda-document-manager')),
        timeout: cdk.Duration.seconds(900),
        memorySize: 8192,
        environment: {
          s3_bucket: s3Bucket.bucketName,
          s3_prefix: s3_prefix,
          s3_capture_prefix: s3_capture_prefix,
          opensearch_url: opensearch_url,
          roleArn: roleLambdaDocument.roleArn,
          path: 'https://'+distribution_sharing.domainName+'/', 
          sqsUrl: queueUrl[i],
          max_object_size: String(max_object_size),
          supportedFormat: supportedFormat,
          LLM_embedding: JSON.stringify(LLM_embedding),
          enableParentDocumentRetrival: enableParentDocumentRetrival,
          enableHybridSearch: enableHybridSearch,
          vectorIndexName: vectorIndexName
        }
      });         
      s3Bucket.grantReadWrite(lambdDocumentManager[i]); // permission for s3
      lambdDocumentManager[i].addEventSource(new SqsEventSource(queue[i])); // permission for SQS
    }

    // Weather
    new secretsmanager.Secret(this, `weather-api-secret-for-${projectName}`, {
      description: 'secret for weather api key', // openweathermap
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      secretName: `openweathermap-${projectName}`,
      secretObjectValue: {
        project_name: cdk.SecretValue.unsafePlainText(projectName),
        weather_api_key: cdk.SecretValue.unsafePlainText(''),
      },
    });

    // Tavily
    new secretsmanager.Secret(this, `tavily-secret-for-${projectName}`, {
      description: 'secret for tavily api key', // tavily
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      secretName: `tavilyapikey-${projectName}`,
      secretObjectValue: {
        project_name: cdk.SecretValue.unsafePlainText(projectName),
        tavily_api_key: cdk.SecretValue.unsafePlainText(''),
      },
    });

    // perplexity
    new secretsmanager.Secret(this, `perflexity-secret-for-${projectName}`, {
      description: 'secret for perflexity api key', // tavily
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      secretName: `perplexityapikey-${projectName}`,
      secretObjectValue: {
        project_name: cdk.SecretValue.unsafePlainText(projectName),
        perplexity_api_key: cdk.SecretValue.unsafePlainText(''),
      },
    });

    // firecrawl
    new secretsmanager.Secret(this, `firecrawl-secret-for-${projectName}`, {
      description: 'secret for firecrawl api key', // firecrawl
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      secretName: `firecrawlapikey-${projectName}`,
      secretObjectValue: {
        project_name: cdk.SecretValue.unsafePlainText(projectName),
        firecrawl_api_key: cdk.SecretValue.unsafePlainText(''),
      },
    });

    // langsmith
    const langsmithApiSecret = new secretsmanager.Secret(this, `langsmith-secret-for-${projectName}`, {
      description: 'secret for lamgsmith api key', // langsmith
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      secretName: `langsmithapikey-${projectName}`,
      secretObjectValue: {
        langchain_project: cdk.SecretValue.unsafePlainText(projectName),
        langsmith_api_key: cdk.SecretValue.unsafePlainText(''),
      }, 
    });

    const lambdaKnowledgeBase = new lambda.DockerImageFunction(this, `knowledge-base-for-${projectName}`, {
      description: 'RAG based on Knoeledge Base',
      functionName: `knowledge-base-for-${projectName}`,
      code: lambda.DockerImageCode.fromImageAsset(path.join(__dirname, '../../lambda-knowledge-base')),
      timeout: cdk.Duration.seconds(120),
      memorySize: 4096,
      role: roleLambdaKnowledgeBase,
      environment: {
        bedrock_region: String(region),  
        projectName: projectName,
        "sharing_url": 'https://'+distribution_sharing.domainName,
      }
    });
    lambdaKnowledgeBase.grantInvoke(new cdk.aws_iam.ServicePrincipal("bedrock.amazonaws.com"));     

    // lambda Opensearch
    const roleLambdaOpenSearch = new iam.Role(this, `role-lambda-opensearch-for-${projectName}`, {
      roleName: `role-lambda-opensearch-for-${projectName}-${region}`,
      assumedBy: new iam.CompositePrincipal(
        new iam.ServicePrincipal("lambda.amazonaws.com"),
        new iam.ServicePrincipal("bedrock.amazonaws.com"),
      ),
    });
    roleLambdaOpenSearch.attachInlinePolicy( 
      new iam.Policy(this, `log-policy-lambda-opensearch-for-${projectName}`, {
        statements: [CreateLogPolicy],
      }),
    );
    roleLambdaOpenSearch.attachInlinePolicy( 
      new iam.Policy(this, `stream-log-policy-lambda-opensearch-for-${projectName}`, {
        statements: [CreateLogStreamPolicy],
      }),
    );      
    roleLambdaOpenSearch.attachInlinePolicy( 
      new iam.Policy(this, `bedrock-invoke-policy-opensearch-for-${projectName}`, {
        statements: [bedrockInvokePolicy],
      }),
    );  
    roleLambdaOpenSearch.attachInlinePolicy(
      new iam.Policy(this, `opensearch-policy-lambda-opensearch-for-${projectName}`, {
        statements: [openSearchPolicy],
      }),
    );    
    roleLambdaOpenSearch.attachInlinePolicy( 
      new iam.Policy(this, `s3-policy-lambda-opensearch-for-${projectName}`, {
        statements: [S3Policy],
      }),
    );    

    // Lambda for opensearch
    const lambdaOpenSearch = new lambda.DockerImageFunction(this, `opensearch-for-${projectName}`, {
      description: 'RAG based on OpenSearch',
      functionName: `opensearch-for-${projectName}`,
      code: lambda.DockerImageCode.fromImageAsset(path.join(__dirname, '../../lambda-opensearch')),
      timeout: cdk.Duration.seconds(120),
      memorySize: 4096,
      role: roleLambdaOpenSearch,
      environment: {
        bedrock_region: String(region),  
        projectName: projectName,
        opensearch_url: opensearch_url,
        "sharing_url": 'https://'+distribution_sharing.domainName,
      }
    });
    lambdaOpenSearch.grantInvoke(new cdk.aws_iam.ServicePrincipal("bedrock.amazonaws.com"));     

    const environment = {
      "projectName": projectName,
      "accountId": accountId,
      "region": region,
      "roleKnowledgeBase": roleKnowledgeBase.roleArn,
      "collectionArn": collectionArn,
      "serverless_opensearch_url": OpenSearchCollection.attrCollectionEndpoint,
      "managed_opensearch_url": opensearch_url,
      "s3_bucket": s3Bucket.bucketName,      
      "s3_arn": s3Bucket.bucketArn,
      "sharing_url": 'https://'+distribution_sharing.domainName,
    }    
    new cdk.CfnOutput(this, `environment-for-${projectName}`, {
      value: JSON.stringify(environment),
      description: `environment-${projectName}`,
      exportName: `environment-${projectName}`
    });    
  }
}
