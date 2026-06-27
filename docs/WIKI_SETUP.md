# Triangle LLM Internal Documentation Wiki Setup

**Confidential - For Authorized Use Only**

## Overview

The Triangle LLM Internal Wiki serves as the central knowledge repository for the Triangle LLM development team. It contains proprietary documentation, best practices, architecture decisions, and operational procedures.

## Wiki Structure

```
triangle-llm-wiki/
├── home.md                          # Wiki homepage
├── getting-started/
│   ├── quickstart.md               # Quick start guide
│   ├── environment-setup.md        # Dev environment setup
│   └── first-contribution.md       # Contribution guide
├── architecture/
│   ├── overview.md                 # System architecture
│   ├── components.md               # Component descriptions
│   ├── adr/                        # Architecture Decision Records
│   │   ├── adr-001-moe.md
│   │   ├── adr-002-attention.md
│   │   └── adr-003-safety.md
│   └── diagrams/                   # Architecture diagrams
├── development/
│   ├── development-guide.md        # Dev workflow
│   ├── testing-guide.md            # Testing procedures
│   ├── debugging.md                # Debugging tips
│   └── performance-tuning.md       # Optimization guide
├── deployment/
│   ├── deployment-guide.md         # Deployment procedures
│   ├── operations-manual.md        # Ops playbook
│   ├── troubleshooting.md          # Troubleshooting guide
│   └── incident-response.md        # Incident procedures
├── security/
│   ├── security-policies.md        # Security policies
│   ├── threat-model.md             # Threat modeling
│   ├── compliance.md               # Compliance checklist
│   └── audit-procedures.md         # Audit procedures
├── finetuning/
│   ├── finetuning-guide.md         # Fine-tuning guide
│   ├── data-preparation.md         # Data prep procedures
│   └── examples.md                 # Fine-tuning examples
├── api-reference/
│   ├── rest-api.md                 # REST API docs
│   ├── python-sdk.md               # Python SDK reference
│   └── examples.md                 # API examples
└── team/
    ├── team-members.md             # Team directory
    ├── oncall-rotation.md          # On-call schedule
    └── meeting-notes/              # Meeting minutes
```

## Setup Instructions

### Step A: Create Wiki Repository

```bash
# Create private wiki repository
WIKI_REPO="TriangleOS/GLM-5-wiki"

curl -X POST \
  -H "Authorization: token ${GH_TOKEN}" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/user/repos \
  -d '{
    "name": "GLM-5-wiki",
    "description": "Triangle LLM Internal Documentation Wiki",
    "private": true,
    "has_issues": false,
    "has_projects": false,
    "has_downloads": false,
    "has_wiki": true,
    "has_pages": true,
    "is_template": false
  }'
```

### Step B: Initialize Wiki Content

```bash
# Clone wiki repository
git clone https://github.com/TriangleOS/GLM-5-wiki.git
cd GLM-5-wiki

# Create wiki structure
mkdir -p getting-started architecture development deployment security finetuning api-reference team

# Initialize pages
touch home.md
touch getting-started/{quickstart,environment-setup,first-contribution}.md
touch architecture/{overview,components}.md
touch development/{development-guide,testing-guide,debugging,performance-tuning}.md
touch deployment/{deployment-guide,operations-manual,troubleshooting,incident-response}.md
touch security/{security-policies,threat-model,compliance,audit-procedures}.md
touch finetuning/{finetuning-guide,data-preparation,examples}.md
touch api-reference/{rest-api,python-sdk,examples}.md
touch team/{team-members,oncall-rotation}.md

# Initialize git
git add .
git commit -m "Initialize Triangle LLM Wiki structure"
git push origin main
```

### Step C: Access Controls

```bash
# Configure team access
curl -X PUT \
  -H "Authorization: token ${GH_TOKEN}" \
  "https://api.github.com/teams/triangle-llm-maintainers/repos/TriangleOS/GLM-5-wiki" \
  -d '{
    "permission": "maintain"
  }'

# Add security team as read-only
curl -X PUT \
  -H "Authorization: token ${GH_TOKEN}" \
  "https://api.github.com/teams/security-team/repos/TriangleOS/GLM-5-wiki" \
  -d '{
    "permission": "pull"
  }'
```

## Key Wiki Pages

### 1. Home Page (home.md)

```markdown
# Triangle LLM Internal Wiki

**Access**: Triangle OS Authorized Users Only

Welcome to the Triangle LLM internal documentation wiki. This is the central hub for all 
proprietary Triangle LLM development information.

## Quick Links

- [Getting Started](getting-started/quickstart.md) - New team member onboarding
- [Architecture Overview](architecture/overview.md) - System design
- [Development Guide](development/development-guide.md) - How to contribute
- [Deployment Guide](deployment/deployment-guide.md) - Production deployment
- [Security Policies](security/security-policies.md) - Security requirements

## Latest Updates

- [2026-06-27] CI/CD Pipeline configured
- [2026-06-27] Branch protection rules enabled
- [2026-06-27] Cryptographic signing implemented

## Quick Links

- [Team Directory](team/team-members.md)
- [On-Call Rotation](team/oncall-rotation.md)
- [Incident Response](deployment/incident-response.md)
```

### 2. Architecture Decision Records (ADRs)

ADRs document important architectural decisions:

```markdown
# ADR-001: Mixture of Experts Architecture

## Status
Accepted (2026-04-15)

## Context
GLM-5 uses a mixture-of-experts (MoE) architecture for scalability.

## Decision
Triangle LLM maintains MoE architecture for maximum compatibility with GLM-5.

## Consequences
- Enables horizontal scaling of model capacity
- Increases model complexity vs dense models
- Allows per-token expert specialization

## Related
- GLM-5 Technical Report
- IndexShare paper
```

### 3. Threat Model (security/threat-model.md)

```markdown
# Triangle LLM Threat Model

## Threat Actors

1. **External Attackers**
   - Attempt unauthorized model access
   - Exploit vulnerabilities for data exfiltration
   - Perform denial-of-service attacks

2. **Insider Threats**
   - Unauthorized model redistribution
   - Data theft via API access
   - Credential compromise

## High-Priority Threats

- Model weight theft
- Unauthorized API access
- Data poisoning attacks
- Prompt injection attacks

## Mitigations

- Cryptographic model verification
- Multi-factor authentication
- Audit logging and monitoring
- Input/output content filtering
- Rate limiting and DDoS protection
```

## Wiki Synchronization

Automatically sync GitHub repository documentation to wiki:

```python
# scripts/sync_wiki.py
#!/usr/bin/env python3

import os
import json
from pathlib import Path
import subprocess

def sync_wiki(docs_path: str, wiki_endpoint: str, api_token: str):
    """Sync docs directory to internal wiki."""
    
    docs_path = Path(docs_path)
    
    for md_file in docs_path.rglob('*.md'):
        # Skip certain files
        if any(x in str(md_file) for x in ['PRIVATE', 'DRAFT']):
            continue
            
        # Read content
        with open(md_file, 'r') as f:
            content = f.read()
        
        # Compute page path
        relative_path = md_file.relative_to(docs_path)
        page_title = relative_path.stem
        
        # Upload to wiki
        upload_to_wiki(
            endpoint=wiki_endpoint,
            token=api_token,
            title=page_title,
            content=content,
            path=str(relative_path)
        )

if __name__ == '__main__':
    sync_wiki(
        docs_path='./docs',
        wiki_endpoint=os.getenv('WIKI_ENDPOINT'),
        api_token=os.getenv('WIKI_API_TOKEN')
    )
```

## Wiki Maintenance

### Access Logging

All wiki access is logged:
```json
{
  "timestamp": "2026-06-27T10:30:45Z",
  "user": "team_member@triangleos.com",
  "action": "view_page",
  "page": "security/threat-model.md",
  "ip_address": "redacted"
}
```

### Version Control

All wiki changes tracked via git:
```bash
# View wiki change history
cd GLM-5-wiki
git log --oneline

# Restore previous version
git checkout <commit-hash> -- architecture/overview.md
```

### Update Schedule

- **Weekly**: Review and update team member access
- **Bi-weekly**: Update meeting notes and decisions
- **Monthly**: Comprehensive documentation review
- **Quarterly**: Architecture documentation refresh

---

**Classification**: CONFIDENTIAL  
**Maintained By**: Triangle LLM Documentation Team  
**Last Updated**: June 27, 2026
