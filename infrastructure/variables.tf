variable "aws_region" {
  description = "AWS region used to deploy project resources"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Project name used for resource tags"
  type        = string
  default     = "aws-health-ai-platform"
}

variable "ecr_repository_name" {
  description = "Name of the ECR repository for the API"
  type        = string
  default     = "aws-health-ai-api"
}

variable "s3_bucket_prefix" {
  description = "Prefix for the project S3 bucket"
  type        = string
  default     = "aws-health-ai-data-"
}

variable "sagemaker_execution_role_name" {
  description = "Name of the SageMaker execution role"
  type        = string
  default     = "aws-health-ai-sagemaker-execution-role"
}