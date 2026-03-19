# Privacy Policy Implementation Notes (KIT Scientific Computing Service)

## 0. Context / Target System
This privacy policy is written for a scientific computing and storage service operated at KIT with:
- Federated authentication via EGI AAI
- Multi-institution access (e.g., Uni Heidelberg, Uni Freiburg, …)
- Job scheduling and accounting via HTCondor; potential execution/connection to GridKa
- Read-only software distribution via CVMFS
- Collection of resource usage metrics (CPU, RAM, runtime, storage usage, job submission metadata)
- Security/system logs retained for less than 90 days
- No general backup/archival guarantee

Tone target: academic-neutral, technical, transparent.

---

## 1. Key Design Principles (Why we wrote it this way)

### 1.1 Separate “website policy” from “compute infrastructure policy”
A scientific computing service processes fundamentally different data than a public website:
- identity attributes (federated)
- account lifecycle data
- job metadata + resource accounting
- security logs for incident response

Therefore we explicitly scope the policy to the Service and its operational components.

### 1.2 Explicitly enumerate “personal data categories”
Even if the Service stores “scientific data” and is not intended for personal data, personal data still exists via:
- identity attributes (name/email/affiliation)
- IP addresses and login events
- job/accounting metadata that can be linked to a user

Enumerating categories is essential for GDPR Art. 13 transparency.

### 1.3 Clarify roles and boundaries (Controller vs external operators)
Because GridKa and EGI federation are involved:
- KIT is controller for operating the Service at KIT
- External infrastructure operators may be controllers for their own logging/accounting
The policy states data may flow to external operators for job execution/accounting and that they have their own responsibility.

### 1.4 Resource usage collection: operational necessity, not profiling
We include resource usage data collection as a core purpose:
- fairness (quotas, allocations)
- stability/capacity planning
- grant/project reporting
But we explicitly disclaim:
- no profiling
- no automated decision-making on individuals beyond operations/security

This reduces risk of “employee monitoring” interpretations and keeps intent aligned with research infra norms.

### 1.5 Retention: minimal, < 90 days for security logs
We fix retention to “< 90 days” for security/system logs (unless incident handling requires longer).
This is consistent with data minimization principles and your operational constraints.

### 1.6 No backup guarantee: explicit statement
Because there is no general backup system:
- We explicitly disclose the absence of backup/archival guarantees
- We place responsibility on users/projects to manage durable copies
This avoids user expectations and liability issues.

---

## 2. Assumptions We Made (Explicit)
The drafted policy assumes:
1. The Service is hosted physically/logically at KIT and operated by KIT personnel.
2. CVMFS is used strictly for read-only software distribution (no research data stored there by the Service).
3. Resource usage accounting is stored at KIT; GridKa maintains its own central logs for GridKa-side execution.
4. No commercial third-party processors are used for storage/compute (no cloud processors); if this changes, recipients/processing agreements must be updated.
5. Security/system logs are retained for less than 90 days under normal operations.
6. The Service is intended for scientific data and is not intended for special categories of personal data (Art. 9 GDPR) unless explicitly authorized under a defined legal framework.

If any of these assumptions change, the policy must be updated.

---

## 3. Why the KIT SCC Website Privacy Policy Is Not Sufficient

The SCC Self-Service Portal privacy policy is designed for:
- website access and server logs for web pages
- session cookies
- contact form / newsletter
- generic “website-only” data processing

It is not sufficient for scientific computing infrastructure because it does not cover:

### 3.1 Federated identity (EGI AAI)
A compute service with EGI federation must describe:
- which identity attributes are received
- how they are used for authorization
- data flow between IdP → federation → service

### 3.2 Account lifecycle and access control
A compute service must disclose:
- local account identifiers
- SSH keys (if used)
- role/group/project allocation data
Website policies typically do not cover these.

### 3.3 Job submission metadata and resource accounting
Compute services process:
- job IDs, runtimes, CPU/RAM/storage usage
- queue/project associations
- accounting metadata
This category is absent from website-focused policies.

### 3.4 Security logging beyond web server logs
Compute security logs include:
- SSH/login events, source IP, failed attempts
- scheduler events
Website policy “7-day anonymized web logs” is not representative.

### 3.5 External infrastructure integration (GridKa, HTCondor)
If jobs can be executed externally or interact with external infra:
- recipients and responsibilities must be disclosed
- data transfers must be transparent
Website policies usually do not model this.

### 3.6 Research data responsibility and “not intended for personal data”
A compute policy should include:
- user responsibility for uploaded content
- constraints on special-category personal data
- statement that KIT does not inspect research content routinely
Again absent in a website policy.

---

## 4. Implementation Checklist (Operational Mapping)

### 4.1 Data inventory mapping (what system component generates which data)
- EGI AAI: identity attributes (name/email/affiliation/unique ID, VO)
- Account management: username, SSH keys, roles/groups
- HTCondor: job submission metadata, accounting metrics
- GridKa: GridKa-side execution logs (external responsibility)
- CVMFS: read-only software distribution (ensure no user data stored)
- System security: SSH/auth logs, firewall logs, incident records (keep <90 days)

### 4.2 Retention configuration
- Confirm log rotation and deletion policies enforce <90 days
- Document incident exception policy (e.g., preserve logs if needed for security investigation)

### 4.3 Publishing and versioning
- Provide a canonical URL for the policy
- Add “last updated” stamp (optional but recommended)
- Keep change history (internal) for audits

---

## 5. What is NOT covered here (deliberately)
This privacy policy is not a Terms of Use / Acceptable Use Policy.
You still need a ToU/AUP that covers:
- permitted use / prohibited activities (e.g., abuse, crypto mining)
- sanctions and account suspension
- user responsibilities for data and security
- liability limitations and availability expectations
