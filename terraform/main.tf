terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
  # Credentials via AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY env vars
  # or EC2 instance profile — never hard-code secrets here.
}

# ---------------------------------------------------------------------------
# S3 artifact bucket
# ---------------------------------------------------------------------------

resource "aws_s3_bucket" "artifacts" {
  bucket = var.artifact_bucket_name

  tags = {
    Project     = "llm-from-scratch"
    Environment = var.environment
  }
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# ---------------------------------------------------------------------------
# Security group for GPU training instances
# ---------------------------------------------------------------------------

resource "aws_security_group" "gpu_training" {
  name        = "llm-gpu-training-${var.environment}"
  description = "Allow SSH and inter-node NCCL traffic for GPU training cluster"
  vpc_id      = var.vpc_id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  ingress {
    description = "NCCL inter-node (all TCP within SG)"
    from_port   = 0
    to_port     = 65535
    protocol    = "tcp"
    self        = true
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Project     = "llm-from-scratch"
    Environment = var.environment
  }
}

# ---------------------------------------------------------------------------
# GPU training EC2 instance (placeholder — use ASG or spot fleet in production)
# ---------------------------------------------------------------------------

resource "aws_instance" "gpu_training" {
  count = var.gpu_instance_count

  ami           = var.gpu_ami_id         # Amazon Linux 2023 Deep Learning AMI
  instance_type = var.gpu_instance_type  # e.g. g5.12xlarge, p4d.24xlarge

  vpc_security_group_ids = [aws_security_group.gpu_training.id]
  subnet_id              = var.subnet_id

  key_name = var.key_pair_name

  root_block_device {
    volume_size = 500   # GB
    volume_type = "gp3"
    encrypted   = true
  }

  iam_instance_profile = aws_iam_instance_profile.gpu_training.name

  tags = {
    Name        = "llm-gpu-training-${var.environment}-${count.index}"
    Project     = "llm-from-scratch"
    Environment = var.environment
  }
}

# ---------------------------------------------------------------------------
# IAM role for S3 access from training instances
# ---------------------------------------------------------------------------

resource "aws_iam_role" "gpu_training" {
  name = "llm-gpu-training-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "s3_access" {
  name = "s3-artifacts-access"
  role = aws_iam_role.gpu_training.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ]
      Resource = [
        aws_s3_bucket.artifacts.arn,
        "${aws_s3_bucket.artifacts.arn}/*"
      ]
    }]
  })
}

resource "aws_iam_instance_profile" "gpu_training" {
  name = "llm-gpu-training-profile-${var.environment}"
  role = aws_iam_role.gpu_training.name
}
