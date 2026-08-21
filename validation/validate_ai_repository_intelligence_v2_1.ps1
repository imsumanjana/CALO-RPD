param(
    [Parameter(Mandatory = $false)]
    [string]$RepositoryRoot = "."
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$root = (Resolve-Path $RepositoryRoot).Path
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logRoot = Join-Path $root ".ai-tmp/validation-$stamp"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$transcript = Join-Path $logRoot "validation.log"
Start-Transcript -Path $transcript -Force | Out-Null

$legacyIndexes = @(
    "file-index.json", "symbol-index.json", "dependency-graph.json",
    "test-map.json", "audit-coverage.json", "change-index.json"
)

function Invoke-Step {
    param([string]$Name, [scriptblock]$Body)
    Write-Host ""
    Write-Host "=== $Name ==="
    $global:LASTEXITCODE = 0
    & $Body
    $code = $global:LASTEXITCODE
    if ($null -ne $code -and $code -ne 0) {
        throw "$Name failed with exit code $code"
    }
}

function Require-Path {
    param([string]$Path)
    if (-not (Test-Path $Path)) { throw "Required v2 artifact is missing: $Path" }
}

function Get-TreeHashes {
    param([string]$Path)
    $base = (Resolve-Path $Path).Path
    $map = @{}
    Get-ChildItem $base -Recurse -File | Sort-Object FullName | ForEach-Object {
        $relative = $_.FullName.Substring($base.Length).TrimStart('\','/') -replace '\\','/'
        $map[$relative] = (Get-FileHash -Algorithm SHA256 $_.FullName).Hash.ToLowerInvariant()
    }
    return $map
}

function Assert-HashMapsEqual {
    param($Before, $After, [string]$Label)
    $keys = @($Before.Keys + $After.Keys | Sort-Object -Unique)
    $changed = @()
    foreach ($key in $keys) {
        if (-not $Before.ContainsKey($key) -or -not $After.ContainsKey($key) -or $Before[$key] -ne $After[$key]) {
            $changed += $key
        }
    }
    if ($changed.Count -gt 0) {
        $changed | Set-Content -Encoding utf8 (Join-Path $logRoot "$Label-changed-files.txt")
        throw "$Label was not byte-stable. Changed files: $($changed -join ', ')"
    }
}

try {
    Set-Location $root

    Invoke-Step "Preflight" {
        Require-Path "scripts/ai-index"
        Require-Path "scripts/ai-agent-guard.py"
        Require-Path ".ai/index/manifest.json"
        Require-Path ".ai/index/change-index.json"
        Require-Path ".ai/architectural-semantics.json"
        Require-Path ".ai/semantic-benchmark.json"
        Require-Path ".ai/audit-seed.json"
        Require-Path "tests/tooling/test_ai_repo_intelligence_v2.py"
        Require-Path ".github/workflows/ai-index.yml"
        git rev-parse --show-toplevel | Set-Content -Encoding utf8 (Join-Path $logRoot "git-root.txt")
        git rev-parse HEAD | Set-Content -Encoding utf8 (Join-Path $logRoot "git-head.txt")
        git branch --show-current | Set-Content -Encoding utf8 (Join-Path $logRoot "git-branch.txt")
        $status = @(git -c core.safecrlf=false status --short --untracked-files=normal)
        $status | Set-Content -Encoding utf8 (Join-Path $logRoot "git-status-before.txt")
        Write-Host "Working-tree status entries logged: $($status.Count)"
        Write-Host "Repository-wide git diff --check intentionally skipped: migration tree is dirty and Windows EOL conversion makes it noisy/slow."
    }

    Invoke-Step "Protected agent policy scope" {
        python scripts/ai-agent-guard.py --check --root .
        $policy = Get-Content ".ai/agent-policy.json" -Raw | ConvertFrom-Json
        $targets = @($policy.agent_files)
        if ($targets.Count -lt 1) { throw "Agent policy has no active targets" }
        foreach ($target in $targets) {
            $normalized = $target -replace '\\','/'
            if ($normalized -match '^(artifacts|validation|publication_export)/' -or
                $normalized -match '^calo_rpd_studio/data/(pglib|trained_models|frozen)/') {
                throw "Agent policy includes an out-of-scope evidence/generated target: $target"
            }
        }
        Write-Host "Active agent-policy targets: $($targets.Count)"
    }

    Invoke-Step "Sharded index freshness" {
        python scripts/ai-index check
        python scripts/ai-index status | Set-Content -Encoding utf8 (Join-Path $logRoot "index-status.txt")
        python scripts/ai-index guard --no-semantic
    }

    Invoke-Step "Canonical sharded storage" {
        $manifest = Get-Content ".ai/index/manifest.json" -Raw | ConvertFrom-Json
        if ($manifest.sharded -ne $true) { throw "manifest.sharded is not true" }
        if ($manifest.coverage_complete -ne $true) { throw "manifest.coverage_complete is not true" }
        if ([int]$manifest.counts.files -lt 1) { throw "Manifest contains no indexed files" }
        if ([int]$manifest.counts.symbols -lt 1) { throw "Manifest contains no indexed symbols" }
        foreach ($name in $legacyIndexes) {
            if (Test-Path (Join-Path ".ai" $name)) { throw "Legacy generated root index must be absent in v2: $name" }
        }
        foreach ($kind in @("files","symbols","dependencies","tests","audit","modules")) {
            Require-Path (Join-Path ".ai/index" $kind)
        }
    }

    Invoke-Step "Migration separated from recent changes" {
        $change = Get-Content ".ai/index/change-index.json" -Raw | ConvertFrom-Json
        if ($change.initialization -eq $true) { throw "Initialization flag is still true after migration/remediation" }
        $changedFiles = @($change.changed_files)
        $newFiles = @($change.newly_indexed_files)
        if ($newFiles.Count -gt 0) {
            throw "Migration artifacts are leaking into ordinary newly-indexed changes: $($newFiles -join ', ')"
        }
        Write-Host "Ordinary recent changed files: $($changedFiles.Count)"
    }

    Invoke-Step "Historical audit preservation" {
        $seed = Get-Content ".ai/audit-seed.json" -Raw | ConvertFrom-Json
        $manifest = Get-Content ".ai/index/manifest.json" -Raw | ConvertFrom-Json
        $units = @($seed.units.PSObject.Properties)
        if ($units.Count -ne 9) { throw "Expected 9 historical audit units; found $($units.Count)" }
        $retained = 0
        foreach ($unitProp in $units) {
            $sourcePath = $unitProp.Name
            $seedRecord = $unitProp.Value
            $manifestProp = $manifest.files.PSObject.Properties | Where-Object { $_.Name -eq $sourcePath } | Select-Object -First 1
            if ($null -eq $manifestProp) { throw "Audited source missing from manifest: $sourcePath" }
            $auditPath = ".ai/index/audit/$($manifestProp.Value.key).json"
            Require-Path $auditPath
            $audit = (Get-Content $auditPath -Raw | ConvertFrom-Json).audit
            if ($audit.reviewed -ne $true) { throw "Historical reviewed state was not retained for $sourcePath" }
            if ($audit.re_audit_required -eq $true) { throw "Unchanged historical audit was incorrectly invalidated for $sourcePath" }
            if ($audit.last_reviewed_content_sha256 -ne $seedRecord.content_hash) {
                throw "Historical audit content identity mismatch for $sourcePath"
            }
            $retained++
        }
        Write-Host "Retained historical audit units: $retained"
    }

    Invoke-Step "Byte-stable regeneration" {
        $before = Get-TreeHashes ".ai"
        python scripts/ai-index init
        $after = Get-TreeHashes ".ai"
        Assert-HashMapsEqual $before $after "ai-init-idempotence"
        python scripts/ai-index check
        foreach ($name in $legacyIndexes) {
            if (Test-Path (Join-Path ".ai" $name)) { throw "Legacy root index was regenerated by init: $name" }
        }
    }

    Invoke-Step "Deterministic no-semantic context" {
        $q = "where should policy registry accounting be modified"
        $one = Join-Path $logRoot "context-1.txt"
        $two = Join-Path $logRoot "context-2.txt"
        python scripts/ai-index context $q --no-semantic | Set-Content -Encoding utf8 $one
        python scripts/ai-index context $q --no-semantic | Set-Content -Encoding utf8 $two
        if ((Get-FileHash -Algorithm SHA256 $one).Hash -ne (Get-FileHash -Algorithm SHA256 $two).Hash) {
            throw "Deterministic context output changed between identical queries"
        }
        $ctx = Get-Content $one -Raw
        if ($ctx -notmatch "policy_registry\.py") { throw "Policy-registry context did not route to the public policy registry" }
        $publicPos = $ctx.IndexOf("policy_registry.py")
        $corePos = $ctx.IndexOf("_policy_registry_core.py")
        if ($corePos -ge 0 -and $corePos -lt $publicPos) { throw "Internal policy core ranked ahead of the public edit surface" }
    }

    Invoke-Step "Stable query API and conservative test mapping" {
        python scripts/ai-index query repo_overview | Set-Content -Encoding utf8 (Join-Path $logRoot "query-repo-overview.json")
        python scripts/ai-index query get_recent_changes | Set-Content -Encoding utf8 (Join-Path $logRoot "query-recent-changes.json")
        python scripts/ai-index query find_symbol PolicyRegistry | Set-Content -Encoding utf8 (Join-Path $logRoot "query-policy-registry.json")
        python scripts/ai-index query get_findings | Set-Content -Encoding utf8 (Join-Path $logRoot "query-findings.json")
        $depsRaw = python scripts/ai-index query get_dependencies calo-policy
        if ($LASTEXITCODE -ne 0) { throw "get_dependencies calo-policy failed" }
        $testsRaw = python scripts/ai-index query get_tests calo-policy
        if ($LASTEXITCODE -ne 0) { throw "get_tests calo-policy failed" }
        $deps = @($depsRaw | ConvertFrom-Json)
        $mappedTests = @($testsRaw | ConvertFrom-Json)
        $badMapped = @($mappedTests | Where-Object {
            $p = $_.ToString().ToLowerInvariant().Replace('\\','/')
            $p.EndsWith('/agents.md') -or $p.EndsWith('/agent.md') -or
            $p.EndsWith('/__init__.py') -or $p.EndsWith('/conftest.py') -or
            $p.EndsWith('/helpers.py')
        })
        if ($badMapped.Count -gt 0) {
            throw "Non-test support/instruction files are mapped as tests: $($badMapped -join ', ')"
        }
        if (-not ($mappedTests | Where-Object { $_ -match '/test_.*\.py$|_test\.[A-Za-z0-9]+$|\.test\.|\.spec\.' })) {
            throw "calo-policy has no executable test-source mapping after conservative filtering"
        }
        Write-Host "calo-policy dependencies: $($deps.Count); mapped executable tests: $($mappedTests.Count)"
    }

    Invoke-Step "Semantic cache and repository benchmark" {
        $oldProvider = $env:AI_INDEX_EMBEDDING_PROVIDER
        try {
            $env:AI_INDEX_EMBEDDING_PROVIDER = "concept"
            python scripts/ai-index embeddings update | Set-Content -Encoding utf8 (Join-Path $logRoot "embeddings-update-1.json")
            python scripts/ai-index embeddings update | Set-Content -Encoding utf8 (Join-Path $logRoot "embeddings-update-2.json")
            python scripts/ai-index embeddings status | Set-Content -Encoding utf8 (Join-Path $logRoot "embeddings-status.json")
            python scripts/ai-index embeddings benchmark --check | Tee-Object -FilePath (Join-Path $logRoot "semantic-benchmark.json")
        }
        finally {
            $env:AI_INDEX_EMBEDDING_PROVIDER = $oldProvider
        }
    }

    Invoke-Step "V2 tooling regression suite" {
        python -m pytest -q tests/tooling/test_ai_repo_intelligence_v2.py
    }

    Invoke-Step "Legacy tooling regression suite" {
        if (Test-Path "tests/tooling/test_ai_repo_intelligence.py") {
            Write-Host "Legacy v1 tooling tests are retained as historical implementation tests, but their monolithic-storage assertions are not part of the v2 storage contract."
            $legacyOutput = @(
                python -m pytest -q tests/tooling/test_ai_repo_intelligence.py -k "not init_is_deterministic_and_maps_dependencies and not incremental_change_marks_prior_audit_for_reaudit and not malformed_python_is_recorded_not_fatal and not update_reparses_only_changed_python_files and not context_is_targeted" 2>&1
            )
            $legacyCode = $LASTEXITCODE
            $legacyOutput | ForEach-Object { Write-Host $_ }
            $legacyOutput | Set-Content -Encoding utf8 (Join-Path $logRoot "legacy-tooling-regression.txt")
            if ($legacyCode -eq 5) {
                $legacyText = ($legacyOutput -join "`n")
                if ($legacyText -notmatch '5 deselected') {
                    throw "Legacy pytest returned no-tests-collected without the expected five explicit v1 deselections"
                }
                Write-Host "All 5 legacy v1 monolithic-storage tests were intentionally deselected; no compatible legacy test remains to execute."
                $global:LASTEXITCODE = 0
            }
            elseif ($legacyCode -ne 0) {
                $global:LASTEXITCODE = $legacyCode
            }
            else {
                $global:LASTEXITCODE = 0
            }
        }
    }

    Invoke-Step "Read-only CI contract" {
        $workflow = Get-Content ".github/workflows/ai-index.yml" -Raw
        if ($workflow -notmatch "contents:\s*read") { throw "AI workflow does not use contents: read" }
        if ($workflow -notmatch "persist-credentials:\s*false") { throw "AI workflow checkout credentials are not disabled" }
        if ($workflow -notmatch "ai-agent-guard\.py --check") { throw "AI workflow does not verify protected agent policy" }
        if ($workflow -notmatch "ai-index guard --no-semantic") { throw "AI workflow does not verify deterministic index freshness" }
        if ($workflow -match "(?m)^\s*git\s+(push|commit)\b") { throw "AI verification workflow contains a write-back git command" }
    }

    git -c core.safecrlf=false status --short --untracked-files=normal | Set-Content -Encoding utf8 (Join-Path $logRoot "git-status-after.txt")

    [pscustomobject]@{
        validator = "ai-repository-intelligence-v2.1"
        completed_at = (Get-Date).ToString("o")
        branch = (git branch --show-current).Trim()
        commit = (git rev-parse HEAD).Trim()
        checks = @(
            "agent-policy-scope", "freshness", "sharded-storage", "legacy-index-absence",
            "migration-recent-change-separation", "audit-preservation", "byte-stable-init",
            "deterministic-context", "sharded-query-api", "conservative-test-mapping",
            "semantic-cache", "semantic-benchmark", "v2-tooling-tests",
            "legacy-v1-tests-intentionally-deselected", "read-only-ci"
        )
        repository_wide_diff_check_skipped_for_dirty_windows_migration = $true
        scientific_runtime_validation_inferred = $false
        human_scientist_acceptance_inferred = $false
    } | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 (Join-Path $logRoot "summary.json")

    Write-Host ""
    Write-Host "VALIDATION PASSED"
    Write-Host "Logs: $logRoot"
    Write-Host "No scientific training, experiment, GUI, or protected-case workload was run."
}
catch {
    Write-Host ""
    Write-Host "VALIDATION FAILED: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Logs: $logRoot"
    throw
}
finally {
    Stop-Transcript | Out-Null
}
