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

variable "skylake_id" {
  description = "Blazar reservation ID for Skylake"
  type        = string
  default     = "1fcec558-6f4d-4263-bf3b-eb06d7b959f7"
}
