module "eks" {
    source = "terraform-aws-modules/eks/aws"
    version = "20.8.4"

    cluster_name = local.cluster_name
    cluster_version =  var.kubernetes_version           # kubernetes version

    subnet_ids = module.vpc.private_subnets

    enable_irsa = true

    tags = {
        cluster = "production"
    }

    vpc_id = module.vpc.vpc_id


    # details about the Nodes handled by the kubernetes cluster
    eks_managed_node_group_defaults = {

        ami_type               = "AL2_x86_64"
        instance_types         = ["t3.medium"]
        vpc_security_group_ids = [aws_security_group.all_worker_mgmt.id]
    }

    # define the details for autoscaling of the nodes
    eks_managed_node_groups = {

        node_group = {
        min_size     = 2        # min number of nodes while autoscaling
        max_size     = 6        # max number of nodes while autoscaling
        desired_size = 2
        }
  }
  


}