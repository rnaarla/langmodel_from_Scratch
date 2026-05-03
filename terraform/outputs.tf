output "artifact_bucket_name" {
  description = "Name of the S3 bucket for model artefacts."
  value       = aws_s3_bucket.artifacts.bucket
}

output "artifact_bucket_arn" {
  description = "ARN of the S3 artefact bucket."
  value       = aws_s3_bucket.artifacts.arn
}

output "gpu_instance_ids" {
  description = "EC2 instance IDs of GPU training nodes."
  value       = aws_instance.gpu_training[*].id
}

output "gpu_instance_private_ips" {
  description = "Private IP addresses of GPU training nodes."
  value       = aws_instance.gpu_training[*].private_ip
}

output "iam_role_arn" {
  description = "ARN of the IAM role attached to GPU training instances."
  value       = aws_iam_role.gpu_training.arn
}
