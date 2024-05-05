# define all the necessary variables here 
variable "vpc_cidr" {
  default = "10.0.0.0/16"
  description = "CIDR range for the vpc"
}

#? how is this defined?? 
#? can we have any random region or current region 
variable "aws_region" {
  default = "eu-north-1"
  description = "The AWS region"
}


variable "kubernetes_version" {
  default = 1.27
  description = "Kubernetes version"
}