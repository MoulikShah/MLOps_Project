output "floating_ips_skylake" {
  description = "Skylake floating IPs"
  value       = openstack_networking_floatingip_v2.floating_ip.address
}

output "skylake_instance_id_node1" {
  description = "Skylake compute instance ID"
  value       = openstack_compute_instance_v2.nodes["node1"].id
}