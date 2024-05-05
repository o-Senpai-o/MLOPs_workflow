provider "aws" {
  region = var.aws_region
}



locals {
  cluster_name = "singh-eks-${custom_string.suffix.result}"
}

resource "custom_string" "suffix" {
    length = 8
    special = false
}



# for getting the availability zones in the crrent region
data "aws_available_zones" "available" {}


# define the VPC here using a predefined module
module "vpc" {
    source = "terraform-aws-modules/vpc/aws"
    version = "5.7.0"
    
    name                    = "singh-eks-vpc"
    cidr                    = var.vpc_cidr
    azs                     = data.aws_available_zones.available.names
    private_subnets         = ["10.0.1.0/24", "10.0.2.0/24"]
    public_subnets          = ["10.0.4.0/24", "10.0.5.0/24"]
    enable_nat_gateway      = true
    single_nat_gateway      = true
    enable_dns_hostnames    = true
    enable_dns_support      = true

    tags = {
        "kubernetes.io/cluster/${local.cluster_name}" = "shared"
    }

    public_subnet_tags = {
        "kubernetes.io/cluster/${local.cluster_name}" = "shared"
        "kubernetes.io/role/elb"                      = "1"

    }

    private_subnet_tags = {
        "kubernetes.io/cluster/${local.cluster_name}" = "shared"
        "kubernetes.io/role/internal-elb"             = "1"

    }


}