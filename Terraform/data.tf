data "openstack_networking_network_v2" "sharednet1" {
  name = "sharednet1"
}
data "openstack_networking_subnet_v2" "sharednet1-subnet" {
  name = "sharednet1-subnet"
}

data "openstack_networking_secgroup_v2" "default_sg" {
  name = "default"
}