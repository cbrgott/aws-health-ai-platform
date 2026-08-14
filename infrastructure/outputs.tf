output "health_ai_data_bucket_name" {
  value = aws_s3_bucket.health_ai_data.bucket
}

output "health_ai_ecr_repository_url" {
  value = aws_ecr_repository.health_ai_api.repository_url
}

output "sagemaker_execution_role_arn" {
  value = aws_iam_role.sagemaker_execution.arn
}