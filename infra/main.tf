# Azure infrastructure for the Government Evidence Agent (Terraform).
#
# Provisions:
#   - Resource Group
#   - Log Analytics Workspace   : telemetry store for App Insights
#   - Application Insights       : monitoring backend
#   - Container Apps Environment : managed runtime for the container
#   - Container App              : the service, autoscaled, with the App Insights
#                                  connection string injected as an env var
#
# Container Apps (rather than AKS) gives scale-to-zero, HTTPS ingress and
# revisions without running a Kubernetes cluster for a single service.

terraform {
  required_version = ">= 1.5"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
  }
}

provider "azurerm" {
  features {}
}

variable "location" {
  type    = string
  default = "uksouth" # UK region — appropriate for a UK gov-style workload.
}

variable "image" {
  type        = string
  description = "Full container image reference, e.g. myacr.azurecr.io/gov-evidence-agent:sha"
}

variable "llm_provider" {
  type    = string
  default = "anthropic" # production uses the frontier model path
}

variable "anthropic_api_key" {
  type      = string
  sensitive = true
  default   = ""
}

resource "azurerm_resource_group" "rg" {
  name     = "rg-gov-evidence-agent"
  location = var.location
}

resource "azurerm_log_analytics_workspace" "law" {
  name                = "law-gov-evidence-agent"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  sku                 = "PerGB2018"
  retention_in_days   = 30
}

resource "azurerm_application_insights" "appi" {
  name                = "appi-gov-evidence-agent"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  workspace_id        = azurerm_log_analytics_workspace.law.id
  application_type    = "web"
}

resource "azurerm_container_app_environment" "env" {
  name                       = "cae-gov-evidence-agent"
  location                   = azurerm_resource_group.rg.location
  resource_group_name        = azurerm_resource_group.rg.name
  log_analytics_workspace_id = azurerm_log_analytics_workspace.law.id
}

resource "azurerm_container_app" "app" {
  name                         = "ca-gov-evidence-agent"
  container_app_environment_id = azurerm_container_app_environment.env.id
  resource_group_name          = azurerm_resource_group.rg.name
  revision_mode                = "Single"

  template {
    min_replicas = 0 # scale to zero when idle -> no cost when unused
    max_replicas = 3 # scale out under load

    container {
      name   = "gov-evidence-agent"
      image  = var.image
      cpu    = 0.5
      memory = "1Gi"

      env {
        name  = "LLM_PROVIDER"
        value = var.llm_provider
      }
      env {
        name  = "ANTHROPIC_API_KEY"
        value = var.anthropic_api_key # injected from a secret in CI, never committed
      }
      env {
        name  = "APPLICATIONINSIGHTS_CONNECTION_STRING"
        value = azurerm_application_insights.appi.connection_string
      }

      liveness_probe {
        transport = "HTTP"
        port      = 8000
        path      = "/health"
      }
      readiness_probe {
        transport = "HTTP"
        port      = 8000
        path      = "/health"
      }
    }
  }

  ingress {
    external_enabled = true
    target_port      = 8000
    traffic_weight {
      percentage      = 100
      latest_revision = true
    }
  }
}

output "app_url" {
  value = "https://${azurerm_container_app.app.ingress[0].fqdn}"
}

output "app_insights_connection_string" {
  value     = azurerm_application_insights.appi.connection_string
  sensitive = true
}
