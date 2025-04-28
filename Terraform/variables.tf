variable "suffix" {
  description = "yva2006"
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