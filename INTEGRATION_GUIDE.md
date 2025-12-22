# 🔍 Devign Vulnerability Scanner for Other Repositories

Hướng dẫn tích hợp Devign Scanner vào **bất kỳ repo C/C++ nào**.

## 🚀 Cách 1: Dùng GitHub Action (Đơn giản nhất)

Thêm file `.github/workflows/security.yml` vào repo của bạn:

```yaml
name: Security Scan

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  vulnerability-scan:
    runs-on: ubuntu-latest
    permissions:
      security-events: write
    
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      
      - name: Devign Vulnerability Scan
        uses: hoangduy0308/C-Vul-Devign@main
        with:
          threshold: '0.5'
          scan-mode: 'diff'        # Chỉ scan file thay đổi
          upload-sarif: 'true'     # Hiển thị trong Security tab
```

### Các options:

| Option | Mô tả | Default |
|--------|-------|---------|
| `path` | Thư mục cần scan | `.` |
| `threshold` | Ngưỡng xác suất (0.0-1.0) | `0.5` |
| `scan-mode` | `diff` (chỉ changed files) hoặc `full` | `diff` |
| `fail-on-findings` | Fail nếu tìm thấy vulnerability | `false` |
| `upload-sarif` | Upload kết quả lên Code Scanning | `true` |

---

## 🐳 Cách 2: Dùng Docker Image

### Trong GitHub Actions:

```yaml
jobs:
  scan:
    runs-on: ubuntu-latest
    container:
      image: ghcr.io/hoangduy0308/devign-scanner:latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Scan
        run: devign-scan scan . -f sarif -o results.sarif
      
      - uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: results.sarif
```

### Chạy local:

```bash
docker run -v $(pwd):/code ghcr.io/hoangduy0308/devign-scanner:latest \
  scan /code -f json
```

---

## 📦 Cách 3: Copy Scanner vào Repo

1. **Download release package:**
   ```bash
   curl -L https://github.com/hoangduy0308/C-Vul-Devign/releases/latest/download/devign-scanner.zip -o devign-scanner.zip
   unzip devign-scanner.zip -d tools/devign
   ```

2. **Thêm vào `.github/workflows/security.yml`:**
   ```yaml
   - name: Install dependencies
     run: pip install torch numpy tqdm --index-url https://download.pytorch.org/whl/cpu
   
   - name: Scan
     run: python tools/devign/devign_scan.py scan src/ -f sarif -o results.sarif
   ```

---

## ⚙️ Cách 4: Tích hợp với các CI khác

### GitLab CI

```yaml
# .gitlab-ci.yml
security-scan:
  image: python:3.10
  stage: test
  script:
    - pip install torch numpy tqdm --index-url https://download.pytorch.org/whl/cpu
    - curl -L $DEVIGN_SCANNER_URL -o scanner.zip && unzip scanner.zip
    - python devign_scan.py scan . -f json -o gl-sast-report.json
  artifacts:
    reports:
      sast: gl-sast-report.json
```

### Jenkins

```groovy
// Jenkinsfile
pipeline {
    agent any
    stages {
        stage('Security Scan') {
            steps {
                sh '''
                    pip install torch numpy tqdm
                    python devign_scan.py scan src/ -f sarif -o results.sarif
                '''
                recordIssues tools: [sarif(pattern: 'results.sarif')]
            }
        }
    }
}
```

### Azure DevOps

```yaml
# azure-pipelines.yml
- task: UsePythonVersion@0
  inputs:
    versionSpec: '3.10'

- script: |
    pip install torch numpy tqdm
    python devign_scan.py scan $(Build.SourcesDirectory) -f sarif -o $(Build.ArtifactStagingDirectory)/results.sarif
  displayName: 'Run Devign Scanner'

- task: PublishBuildArtifacts@1
  inputs:
    pathToPublish: '$(Build.ArtifactStagingDirectory)/results.sarif'
```

---

## 📋 Ví dụ Workflow Đầy Đủ

```yaml
# .github/workflows/security.yml
name: Security Scan

on:
  push:
    branches: [main, develop]
    paths: ['**.c', '**.h', '**.cpp', '**.hpp']
  pull_request:
    paths: ['**.c', '**.h', '**.cpp', '**.hpp']
  schedule:
    - cron: '0 2 * * 1'  # Weekly full scan

jobs:
  devign-scan:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      security-events: write
      pull-requests: write
    
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      
      - name: Devign Vulnerability Scan
        id: scan
        uses: hoangduy0308/C-Vul-Devign@main
        with:
          threshold: '0.5'
          scan-mode: ${{ github.event_name == 'schedule' && 'full' || 'diff' }}
          fail-on-findings: 'false'
      
      - name: Comment on PR
        if: github.event_name == 'pull_request' && steps.scan.outputs.findings-count > 0
        uses: actions/github-script@v7
        with:
          script: |
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `⚠️ **Devign Scanner** found ${{ steps.scan.outputs.findings-count }} potential vulnerabilities.\n\nPlease check the Security tab for details.`
            })
```

---

## 🔧 Tùy chỉnh Threshold theo Project

| Project Type | Recommended Threshold |
|--------------|----------------------|
| Production code | 0.5 (balanced) |
| Security-critical | 0.3 (more sensitive) |
| Legacy code | 0.7 (reduce noise) |
| New development | 0.4 (catch early) |

---

## ❓ FAQ

**Q: Tốn bao lâu để scan?**
- ~1-2 giây/file trên CPU
- Diff mode thường < 30 giây

**Q: Có false positives không?**
- Có, như mọi SAST tool. Dùng threshold cao hơn để giảm.

**Q: Hỗ trợ ngôn ngữ nào?**
- C và C++ (.c, .h, .cpp, .hpp, .cc, .cxx)

**Q: Cần GPU không?**
- Không, chạy tốt trên CPU. GPU chỉ giúp nhanh hơn ~2x.
