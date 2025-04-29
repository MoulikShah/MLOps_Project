variable "suffix" {
  description = "Net ID"
  type        = string
  nullable = false
}

variable "key" {
  description = "Name of key pair"
  type        = string
  default     = "final"
}

variable "nodes" {
  type = map(string)
  default = {
    "node1" = "192.168.1.11"
  }
}

variable "skylake_id" {
  description = "Skylake reservation"
  type        = string
}

variable "flavor_skylake" {
  description = "Skylake flavor"
  type        = string
  default     = "baremetal"
}