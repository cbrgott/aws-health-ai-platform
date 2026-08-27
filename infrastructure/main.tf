resource "aws_s3_bucket" "health_ai_data" {
  bucket_prefix = var.s3_bucket_prefix

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

resource "aws_ecr_repository" "health_ai_api" {
  name                 = var.ecr_repository_name
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration {
    scan_on_push = true
  }
  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}
resource "aws_iam_role" "sagemaker_execution" {
  name = var.sagemaker_execution_role_name

  assume_role_policy = jsonencode({
    Version = "2012-10-17"

    Statement = [
      {
        Effect = "Allow"

        Principal = {
          Service = "sagemaker.amazonaws.com"
        }

        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

resource "aws_iam_role_policy" "sagemaker_s3_access" {
  name = "aws-health-ai-sagemaker-s3-access"
  role = aws_iam_role.sagemaker_execution.id

  policy = jsonencode({
    Version = "2012-10-17"

    Statement = [
      {
        Effect = "Allow"

        Action = [
          "s3:ListBucket"
        ]

        Resource = aws_s3_bucket.health_ai_data.arn
      },
      {
        Effect = "Allow"

        Action = [
          "s3:GetObject",
          "s3:PutObject"
        ]

        Resource = "${aws_s3_bucket.health_ai_data.arn}/*"
      }
    ]
  })
}

resource "aws_iam_role_policy" "sagemaker_cloudwatch_logs" {
  name = "aws-health-ai-sagemaker-cloudwatch-logs"
  role = aws_iam_role.sagemaker_execution.id

  policy = jsonencode({
    Version = "2012-10-17"

    Statement = [
      {
        Effect = "Allow"

        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]

        Resource = "*"
      }
    ]
  })
}
resource "aws_iam_role_policy" "sagemaker_pass_role" {
  name = "aws-health-ai-sagemaker-pass-role"
  role = aws_iam_role.sagemaker_execution.id

  policy = jsonencode({
    Version = "2012-10-17"

    Statement = [
      {
        Effect = "Allow"

        Action = [
          "iam:PassRole"
        ]

        Resource = aws_iam_role.sagemaker_execution.arn

        Condition = {
          StringEquals = {
            "iam:PassedToService" = "sagemaker.amazonaws.com"
          }
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "sagemaker_pipeline_actions" {
  name = "aws-health-ai-sagemaker-pipeline-actions"
  role = aws_iam_role.sagemaker_execution.id

  policy = jsonencode({
    Version = "2012-10-17"

    Statement = [
      {
        Effect = "Allow"

        Action = [
          "sagemaker:AddTags",

          "sagemaker:CreateProcessingJob",
          "sagemaker:DescribeProcessingJob",
          "sagemaker:StopProcessingJob",

          "sagemaker:CreateTrainingJob",
          "sagemaker:DescribeTrainingJob",
          "sagemaker:StopTrainingJob",

          "sagemaker:CreateModelPackage",
          "sagemaker:DescribeModelPackage",

          "sagemaker:CreateModelPackageGroup",
          "sagemaker:DescribeModelPackageGroup"
        ]

        Resource = "*"
      }
    ]
  })
}