resource "aws_s3_bucket" "health_ai_data" {
  bucket_prefix = "aws-health-ai-data-"

  tags = {
    Project     = "aws-health-ai-platform"
    Environment = "dev"
    ManagedBy   = "terraform"
  }
}

resource "aws_ecr_repository" "health_ai_api" {
  name                 = "aws-health-ai-api"
  image_tag_mutability = "MUTABLE"
  image_scanning_configuration {
    scan_on_push = true
  }
  tags = {
    Project     = "aws-health-ai-platform"
    Environment = "dev"
    ManagedBy   = "terraform"
  }
}
resource "aws_iam_role" "sagemaker_execution" {
  name = "aws-health-ai-sagemaker-execution-role"

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
    Project     = "aws-health-ai-platform"
    Environment = "dev"
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