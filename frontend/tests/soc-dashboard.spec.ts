import { test, expect } from '@playwright/test';

// Deterministic API mock fixture data
const mockStatus = {
  status: "ONLINE",
  version: "1.0.0",
  current_round: 1,
  rounds_completed: 1,
  accuracy: 0.9990,
  macro_f1: 0.9870,
  peak_ram_gb: 7.01,
  ram_limit_gb: 14.0,
  ram_safe: true,
  algorithm: "Sub-Ensemble Margin Sharing (Custom FL XGBoost)",
  tls_enabled: true,
  aes_mode: "AES-256-GCM",
  hmac_mode: "HMAC-SHA-256",
  timestamp: "2026-10-11T20:00:00Z"
};

const mockClients = [
  {
    client_id: "client_1_host",
    display_name: "Central Coordinator Worker",
    ip_address: "127.0.0.1",
    auth_status: "AUTHENTICATED",
    connection_state: "CONNECTED",
    last_seen: "2026-10-11T20:00:00Z",
    current_round: 1,
    training_status: "IDLE",
    last_submission_result: "ACCEPTED",
    accepted_updates_count: 3,
    rejected_updates_count: 0,
    last_error: null
  },
  {
    client_id: "client_2_friend",
    display_name: "Remote Laptop Client (Friend)",
    ip_address: "192.168.1.105",
    auth_status: "AUTHENTICATED",
    connection_state: "CONNECTED",
    last_seen: "2026-10-11T20:00:00Z",
    current_round: 1,
    training_status: "IDLE",
    last_submission_result: "ACCEPTED",
    accepted_updates_count: 2,
    rejected_updates_count: 0,
    last_error: null
  }
];

const mockEvents = [
  {
    event_id: "evt-001",
    timestamp: "2026-10-11T20:00:00Z",
    client_id: "client_2_friend",
    round_id: 1,
    security_layer: "HMAC",
    event_type: "hmac_verification_passed",
    outcome: "PASS",
    severity: "INFO",
    correlation_id: "corr-111",
    details: { msg: "Cryptographic signature validated successfully" }
  },
  {
    event_id: "evt-002",
    timestamp: "2026-10-11T20:01:00Z",
    client_id: "unauthorized_probe",
    round_id: 1,
    security_layer: "REPLAY_CACHE",
    event_type: "replay_submission_rejected",
    outcome: "REJECTED",
    severity: "CRITICAL",
    correlation_id: "corr-222",
    details: { error: "Duplicate nonce detected in sliding window cache" }
  }
];

const mockHealth = {
  status: "HEALTHY",
  cpu_percent: 18.5,
  cpu_usage_pct: 18.5,
  process_ram_gb: 1.42,
  ram_usage_gb: 1.42,
  system_ram_gb: 6.85,
  ram_limit_gb: 14.0,
  ram_safe: true,
  worker_count: 2,
  active_rounds: 1,
  disk_free_gb: 120.4,
  timestamp: "2026-10-11T20:00:00Z"
};

const mockRounds = [
  {
    round_id: 1,
    status: "COMPLETED",
    required_clients: 2,
    participating_clients: ["client_1_host", "client_2_friend"],
    created_at: "2026-10-11T19:50:00Z",
    completed_at: "2026-10-11T20:00:00Z",
    aggregation_status: "COMPLETED",
    metrics: { accuracy: 0.9990, macro_f1: 0.9870 }
  }
];

const defaultHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Content-Type': 'application/json'
};

async function setupApiMocks(page: any) {
  await page.route(/.*\/api\/status.*/, (route: any) =>
    route.fulfill({ status: 200, headers: defaultHeaders, json: mockStatus })
  );
  await page.route(/.*\/api\/clients.*/, (route: any) =>
    route.fulfill({ status: 200, headers: defaultHeaders, json: mockClients })
  );
  await page.route(/.*\/api\/security\/events.*/, (route: any) =>
    route.fulfill({ status: 200, headers: defaultHeaders, json: mockEvents })
  );
  await page.route(/.*\/api\/system\/health.*/, (route: any) =>
    route.fulfill({ status: 200, headers: defaultHeaders, json: mockHealth })
  );
  await page.route(/.*\/api\/fl\/rounds.*/, (route: any) =>
    route.fulfill({ status: 200, headers: defaultHeaders, json: mockRounds })
  );
  await page.route(/.*\/api\/audit\/trail.*/, (route: any) =>
    route.fulfill({ status: 200, headers: defaultHeaders, json: mockEvents })
  );
}

test.describe('TrustGuard-FL SOC Desktop Interface E2E Suite', () => {

  test('1. Application starts and renders branding and 8-tab navigation shell', async ({ page }) => {
    await setupApiMocks(page);
    await page.goto('/');
    await expect(page.locator('.soc-brand')).toContainText('TrustGuard-FL SOC');

    const navTabs = [
      '#nav-overview',
      '#nav-clients',
      '#nav-fl',
      '#nav-telemetry',
      '#nav-audit',
      '#nav-analytics',
      '#nav-health',
      '#nav-settings'
    ];
    for (const tabId of navTabs) {
      await expect(page.locator(tabId)).toBeVisible();
    }
  });

  test('2. Disconnected banner is displayed when coordinator is offline', async ({ page }) => {
    // Intercept API routes with network failure
    await page.route(/.*\/api\/.*/, (route: any) => route.abort());

    await page.goto('/');
    await expect(page.locator('.disconnect-banner')).toBeVisible();
    await expect(page.locator('.disconnect-banner')).toContainText('Coordinator Offline');
    await expect(page.locator('.soc-header-left')).toContainText('DISCONNECTED');
  });

  test('3. Overview tab renders real operational KPI metrics and server state', async ({ page }) => {
    await setupApiMocks(page);
    await page.goto('/');

    await expect(page.locator('.soc-header-left')).toContainText('ONLINE');

    // Check KPIs
    await expect(page.locator('text=99.90%')).toBeVisible();
    await expect(page.locator('text=0.9870')).toBeVisible();
    await expect(page.locator('text=7.01 GB')).toBeVisible();
    await expect(page.locator('text=Sub-Ensemble Margin Sharing')).toBeVisible();
  });

  test('4. Live Clients inventory displays nodes and supports search filtering', async ({ page }) => {
    await setupApiMocks(page);
    await page.goto('/');
    await page.click('#nav-clients');

    await expect(page.locator('text=client_1_host')).toBeVisible();
    await expect(page.locator('text=client_2_friend')).toBeVisible();

    // Type filter
    await page.fill('#client-search', 'friend');
    await expect(page.locator('text=client_2_friend')).toBeVisible();
    await expect(page.locator('text=client_1_host')).not.toBeVisible();
  });

  test('5. Federated Learning tab shows 14-stage lifecycle pipeline', async ({ page }) => {
    await setupApiMocks(page);
    await page.goto('/');
    await page.click('#nav-fl');

    await expect(page.locator('text=14-Stage Cryptographic Update Lifecycle')).toBeVisible();
    await expect(page.getByText('STAGE 1', { exact: true })).toBeVisible();
    await expect(page.getByText('STAGE 5', { exact: true })).toBeVisible();
    await expect(page.getByText('STAGE 14', { exact: true })).toBeVisible();
  });

  test('6. Security Telemetry page renders interactive cryptographic sandboxes', async ({ page }) => {
    await setupApiMocks(page);
    await page.goto('/');
    await page.click('#nav-telemetry');

    await expect(page.locator('text=Security Telemetry & Interactive Cryptographic Sandbox')).toBeVisible();
    await expect(page.locator('text=AES-256-GCM Encryption')).toBeVisible();
    await expect(page.locator('text=HMAC-SHA-256 Tamper Rejection')).toBeVisible();
    await expect(page.locator('text=Isolation Forest Anomaly Filter')).toBeVisible();
  });

  test('7. Audit Trail page renders persistent JSONL records with inspection', async ({ page }) => {
    await setupApiMocks(page);
    await page.goto('/');
    await page.click('#nav-audit');

    await expect(page.locator('text=Structured Persistent Audit Ledger')).toBeVisible();
    await expect(page.locator('text=evt-001')).toBeVisible();
    await expect(page.locator('text=evt-002')).toBeVisible();
  });

  test('8. Model Analytics page renders confusion matrix and performance benchmarks', async ({ page }) => {
    await setupApiMocks(page);
    await page.goto('/');
    await page.click('#nav-analytics');

    await expect(page.locator('text=Model Analytics & Live Inference Engine')).toBeVisible();
    await expect(page.locator('text=CICIoT2023 (21 Classes)')).toBeVisible();
    await expect(page.locator('text=Test Accuracy')).toBeVisible();
    await expect(page.locator('text=Macro F1 Score')).toBeVisible();
  });

  test('9. System Health page displays CPU, RAM ceiling, and process telemetry', async ({ page }) => {
    await setupApiMocks(page);
    await page.goto('/');
    await page.click('#nav-health');

    await expect(page.locator('text=Coordinator Health & Resource Constraints')).toBeVisible();
    await expect(page.locator('text=Host CPU Usage')).toBeVisible();
    await expect(page.locator('text=14.0 GB Hard Limit')).toBeVisible();
  });

  test('10. Settings page displays network parameters and allows API base configuration', async ({ page }) => {
    await setupApiMocks(page);
    await page.goto('/');
    await page.click('#nav-settings');

    await expect(page.locator('text=Central Coordinator API Configuration')).toBeVisible();
    await expect(page.locator('text=Multi-Device Wi-Fi Launch Instructions')).toBeVisible();
    await expect(page.locator('#api-base-input')).toBeVisible();
  });

  test('11. Responsive mobile viewport layout operates without crashing', async ({ page }) => {
    await setupApiMocks(page);
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');
    await expect(page.locator('.soc-brand')).toBeVisible();
    await page.click('#nav-overview');
    await expect(page.locator('.soc-content')).toBeVisible();
  });

});
