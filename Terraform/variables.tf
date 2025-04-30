variable "suffix" {
  description = "Net ID"
  type        = string
  nullable = false
}

variable "key" {
  description = "Name of key pair"
  type        = string
  default     = "default"
}

variable "nodes" {
  type = map(string)
  default = {
    "node1" = "192.168.1.11"
  }
}