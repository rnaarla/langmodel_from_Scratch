variable "aws_region" {
  description = "AWS region for all resources."
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)."
  type        = string
  default     = "dev"
}

variable "artifact_bucket_name" {
  description = "Globally unique S3 bucket name for model artifacts and datasets."
  type        = string
  # Override via: terraform apply -var='artifact_bucket_name=my-llm-artifacts-2024'
}

variable "gpu_instance_type" {
  description = "EC2 instance type for GPU training nodes."
  type        = string
  default     = "g5.12xlarge"  # 4× A10G GPUs, 96 vCPUs
}

variable "gpu_instance_count" {
  description = "Number of GPU training instances to launch."
  type        = number
  default     = 1
}

variable "gpu_ami_id" {
  description = "AMI ID for the GPU training instances (e.g., AWS Deep Learning AMI)."
  type        = string
  # Find latest: aws ssm get-parameter --name /aws/service/deeplearning/ami/x86_64/pytorch/...
}

variable "vpc_id" {
  description = "VPC ID for training infrastructure."
  type        = string
}

variable "subnet_id" {
  description = "Subnet ID for training instances."
  type        = string
}

variable "key_pair_name" {
  description = "EC2 key pair name for SSH access."
  type        = string
  default     = ""
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to SSH into training instances."
  type        = list(string)
  default     = []
}
