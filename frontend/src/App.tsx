import React, { useState, useEffect } from 'react';
import {
  Shield, Activity, Cpu, FileText, Database,
  Lock, AlertTriangle, List, LayoutDashboard, Server, Settings, Upload, CheckCircle, XCircle
} from 'lucide-react';
import * as api from './api';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, BarChart, Bar
} from 'recharts';

function Badge({ type, children }: { type: 'success' | 'danger' | 'warning' | 'info', children: React.ReactNode }) {
  return <span className={`badge ${type}`}>{children}</span>;
}

// ----------------- PAGES ----------------- //

function OverviewPage() {
  const [status, setStatus] = useState<any>(null);

  useEffect(() => {
    api.fetchStatus().then(setStatus).catch(console.error);
  }, []);

  if (!status) return <div>Loading...</div>;

  return (
    <div className="gap-24">
      <h2>System Overview</h2>

      <div className="grid-3">
        <div className="card">
          <div className="card-header"><span className="card-title">FL Engine</span> <Activity size={18} className="text-info" /></div>
          <div className="card-body">
            <h1 style={{ fontSize: 24, marginBottom: 8 }}>{status.status}</h1>
            <Badge type="success">Active</Badge>
          </div>
        </div>

        <div className="card">
          <div className="card-header"><span className="card-title">Global Model</span> <Database size={18} className="text-info" /></div>
          <div className="card-body">
            <h1 style={{ fontSize: 24, marginBottom: 8 }}>{status.global_model}</h1>
            <div className="text-muted">Rows trained: {status.dataset_rows.toLocaleString()}</div>
          </div>
        </div>

        <div className="card">
          <div className="card-header"><span className="card-title">Test Accuracy</span> <CheckCircle size={18} className="text-success" /></div>
          <div className="card-body">
            <h1 style={{ fontSize: 24, marginBottom: 8 }}>{(status.accuracy * 100).toFixed(2)}%</h1>
            <div className="text-muted">Macro F1: {status.macro_f1.toFixed(4)}</div>
          </div>
        </div>
      </div>

      <div className="grid-2">
        <div className="card">
          <div className="card-header"><span className="card-title">Security Pipeline Status</span> <Shield size={18} /></div>
          <div className="card-body">
            <table style={{ width: '100%' }}>
              <tbody>
                <tr><td>TLS Transport Layer</td><td style={{ textAlign: 'right' }}><Badge type={status.tls === 'ACTIVE' ? 'success' : 'danger'}>{status.tls}</Badge></td></tr>
                <tr><td>AES Encryption</td><td style={{ textAlign: 'right' }}><Badge type={status.aes === 'ACTIVE' ? 'success' : 'danger'}>{status.aes}</Badge></td></tr>
                <tr><td>HMAC Integrity</td><td style={{ textAlign: 'right' }}><Badge type={status.hmac === 'ACTIVE' ? 'success' : 'danger'}>{status.hmac}</Badge></td></tr>
                <tr><td>Isolation Forest</td><td style={{ textAlign: 'right' }}><Badge type={status.isolation_forest === 'ACTIVE' ? 'success' : 'danger'}>{status.isolation_forest}</Badge></td></tr>
              </tbody>
            </table>
          </div>
        </div>

        <div className="card">
          <div className="card-header"><span className="card-title">Resource Constraints</span> <Cpu size={18} /></div>
          <div className="card-body gap-16">
            <div>
              <div className="flex-between" style={{ marginBottom: 8 }}>
                <span>Peak RAM Usage</span>
                <span className="mono">{status.peak_ram_gb.toFixed(2)} GB / {status.ram_limit_gb} GB</span>
              </div>
              <div style={{ height: 8, background: '#27272a', borderRadius: 4, overflow: 'hidden' }}>
                <div style={{ width: `${(status.peak_ram_gb / status.ram_limit_gb) * 100}%`, height: '100%', background: 'var(--color-info)' }}></div>
              </div>
            </div>
            <div>
              <div className="flex-between">
                <span>GPU Acceleration</span>
                <Badge type="success">{status.gpu}</Badge>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function SecurityDemoPage() {
  const [aesPayload, setAesPayload] = useState('XGBoost Booster Payload {...}');
  const [aesResult, setAesResult] = useState<any>(null);

  const [hmacPayload, setHmacPayload] = useState('Federated Update Tree Weights');
  const [hmacResult, setHmacResult] = useState<any>(null);

  const [ifResult, setIfResult] = useState<any>(null);

  return (
    <div className="gap-24">
      <h2>Security Demonstration</h2>
      <div className="text-muted" style={{ maxWidth: 800 }}>
        Use these controls to interactively demonstrate the underlying security cryptography and anomaly detection to mentors without triggering a full FL retraining round.
      </div>

      <div className="card">
        <div className="card-header"><span className="card-title">1. AES-256-GCM Encryption Demo</span> <Lock size={18} /></div>
        <div className="card-body gap-16">
          <div className="flex-row" style={{ gap: 16 }}>
            <input
              type="text"
              value={aesPayload}
              onChange={e => setAesPayload(e.target.value)}
              style={{ flex: 1, padding: '8px 12px', background: '#000', border: '1px solid var(--border-color)', color: '#fff', borderRadius: 6 }}
            />
            <button className="btn btn-primary" onClick={async () => setAesResult(await api.demoAes(aesPayload))}>
              Encrypt & Decrypt
            </button>
          </div>
          {aesResult && (
            <pre>
              [CLIENT] Original Payload: {aesResult.original}{'\n'}
              [CLIENT] AES Encrypting...{'\n'}
              [NETWORK] Ciphertext (Hex): {aesResult.ciphertext_hex}{'\n'}
              [NETWORK] Nonce (Hex): {aesResult.nonce_hex}{'\n'}
              [SERVER] AES Decrypting...{'\n'}
              [SERVER] Decrypted Payload: {aesResult.decrypted}{'\n'}
              Status: {aesResult.status}
            </pre>
          )}
        </div>
      </div>

      <div className="card">
        <div className="card-header"><span className="card-title">2. HMAC-SHA-256 Authentication & Tampering Demo</span> <Shield size={18} /></div>
        <div className="card-body gap-16">
          <div className="flex-row" style={{ gap: 16 }}>
            <input
              type="text"
              value={hmacPayload}
              onChange={e => setHmacPayload(e.target.value)}
              style={{ flex: 1, padding: '8px 12px', background: '#000', border: '1px solid var(--border-color)', color: '#fff', borderRadius: 6 }}
            />
            <button className="btn btn-primary" onClick={async () => setHmacResult(await api.demoHmac(hmacPayload))}>
              Generate & Verify HMAC
            </button>
          </div>
          {hmacResult && (
            <pre>
              [CLIENT] Payload: {hmacResult.payload}{'\n'}
              [CLIENT] HMAC Signature: {hmacResult.signature_hex}{'\n'}
              {'\n'}
              --- Normal Verification ---{'\n'}
              [SERVER] Verifying original payload... MATCH = {hmacResult.verify_original ? 'TRUE' : 'FALSE'}{'\n'}
              {'\n'}
              --- Tamper Test (Simulating MITM attack modifying ciphertext) ---{'\n'}
              [SERVER] Verifying tampered payload... MATCH = {hmacResult.verify_tampered ? 'TRUE' : 'FALSE'}{'\n'}
              {hmacResult.tamper_detected && <span className="text-danger">&gt;&gt; TAMPERING DETECTED! Update Rejected.</span>}
            </pre>
          )}
        </div>
      </div>

      <div className="card">
        <div className="card-header"><span className="card-title">3. Isolation Forest Anomaly Detection Demo</span> <AlertTriangle size={18} /></div>
        <div className="card-body gap-16">
          <button className="btn btn-primary" style={{ alignSelf: 'flex-start' }} onClick={async () => setIfResult(await api.demoIsolationForest())}>
            Run Model Poisoning Simulation
          </button>
          {ifResult && (
            <pre>
              [SIMULATION] Creating a normal client model update...{'\n'}
              [SERVER] Isolation Forest Score: {ifResult.normal_client_score.toFixed(3)}{'\n'}
              [SERVER] Normal Model Accepted: {ifResult.normal_client_accepted ? 'TRUE' : 'FALSE'}{'\n'}
              {'\n'}
              [SIMULATION] Creating a POISONED client model (attacker drastically biases feature 19)...{'\n'}
              [SERVER] Isolation Forest Score: {ifResult.poisoned_client_score.toFixed(3)}{'\n'}
              [SERVER] Poisoned Model Accepted: {ifResult.poisoned_client_accepted ? 'TRUE' : 'FALSE'}{'\n'}
              {!ifResult.poisoned_client_accepted && <span className="text-danger">&gt;&gt; ANOMALY DETECTED! Poisoned model dropped from aggregation.</span>}
            </pre>
          )}
        </div>
      </div>
    </div>
  );
}

function FederatedLearningPage() {
  const [rounds, setRounds] = useState<any[]>([]);
  useEffect(() => { api.fetchRounds().then(setRounds); }, []);

  return (
    <div className="gap-24">
      <h2>Federated Learning Operations</h2>
      <div className="card">
        <div className="card-header"><span className="card-title">FL Round Metrics</span></div>
        <div className="card-body" style={{ height: 300 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={rounds}>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
              <XAxis dataKey="round" stroke="#a1a1aa" />
              <YAxis stroke="#a1a1aa" domain={[0.9, 1.0]} />
              <RechartsTooltip contentStyle={{ backgroundColor: '#18181b', borderColor: '#27272a' }} />
              <Line type="monotone" dataKey="val_acc" stroke="#10b981" name="Validation Accuracy" strokeWidth={2} />
              <Line type="monotone" dataKey="val_macro_f1" stroke="#3b82f6" name="Macro F1" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card">
        <div className="card-header"><span className="card-title">Aggregation Logs</span></div>
        <div className="table-container">
          <table>
            <thead>
              <tr><th>Round</th><th>Accuracy</th><th>Macro F1</th><th>Status</th></tr>
            </thead>
            <tbody>
              {rounds.map(r => (
                <tr key={r.round}>
                  <td>Round {r.round}</td>
                  <td>{(r.val_acc * 100).toFixed(2)}%</td>
                  <td>{r.val_macro_f1.toFixed(4)}</td>
                  <td><Badge type="success">Aggregated</Badge></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function ValidationPage() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true); setError('');
    try {
      const res = await api.predictCsv(file);
      setResult(res);
    } catch (err: any) {
      setError(err.message);
    }
    setLoading(false);
  };

  return (
    <div className="gap-24">
      <h2>CSV Validation & Inference</h2>
      <div className="card">
        <div className="card-body gap-16">
          <div className="text-muted">
            <AlertTriangle size={16} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 4 }} />
            Uploaded files are used only for temporary model validation in memory and are not stored or added to the federated training set.
          </div>

          <div className="flex-row" style={{ gap: 16 }}>
            <input
              type="file"
              accept=".csv"
              onChange={e => setFile(e.target.files?.[0] || null)}
              style={{ flex: 1, padding: 8, background: '#000', border: '1px dashed var(--border-color)', borderRadius: 6 }}
            />
            <button className="btn btn-primary" onClick={handleUpload} disabled={!file || loading}>
              {loading ? 'Processing...' : 'Run Inference'}
            </button>
          </div>

          {error && <div className="text-danger">{error}</div>}
        </div>
      </div>

      {result && (
        <div className="grid-2">
          <div className="card">
            <div className="card-header"><span className="card-title">Validation Summary</span></div>
            <div className="card-body">
              <table style={{ width: '100%' }}>
                <tbody>
                  <tr><td>File Processed</td><td className="mono">{result.filename}</td></tr>
                  <tr><td>Total Rows</td><td className="mono">{result.total_rows_processed}</td></tr>
                  <tr><td>Features Found</td><td className="mono">{result.features_detected}</td></tr>
                  <tr><td>Global Model</td><td><Badge type="success">FL XGBoost Engine</Badge></td></tr>
                </tbody>
              </table>
            </div>
          </div>

          <div className="card" style={{ gridColumn: '1 / -1' }}>
            <div className="card-header"><span className="card-title">Sample Predictions (Top 100)</span></div>
            <div className="table-container">
              <table>
                <thead>
                  <tr><th>Row Index</th><th>Predicted Class</th><th>Confidence</th></tr>
                </thead>
                <tbody>
                  {result.sample_predictions.map((p: any) => (
                    <tr key={p.index}>
                      <td>{p.index}</td>
                      <td>{p.prediction}</td>
                      <td>
                        <div className="flex-row" style={{ gap: 8 }}>
                          <span style={{ width: 40 }}>{(p.confidence * 100).toFixed(1)}%</span>
                          <div style={{ height: 6, width: 100, background: '#27272a', borderRadius: 3 }}>
                            <div style={{ height: '100%', width: `${p.confidence * 100}%`, background: p.confidence > 0.9 ? 'var(--color-success)' : 'var(--color-warning)' }}></div>
                          </div>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}


function ClientsPage() {
  const [clients, setClients] = useState<any[]>([]);
  useEffect(() => { api.fetchClients().then(setClients).catch(console.error); }, []);
  return (
    <div className="gap-24">
      <h2>Connected Clients</h2>
      <div className="card">
        <div className="table-container">
          <table>
            <thead><tr><th>Client ID</th><th>Status</th><th>Last Update</th><th>Security</th><th>Transport</th></tr></thead>
            <tbody>
              {clients.map(c => (
                <tr key={c.id}>
                  <td>{c.id}</td>
                  <td><Badge type={c.status === 'ACTIVE' ? 'success' : 'warning'}>{c.status}</Badge></td>
                  <td>{c.last_update}</td>
                  <td><Badge type="success">{c.security}</Badge></td>
                  <td>{c.tls}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function AlertsPage() {
  const [events, setEvents] = useState<any[]>([]);
  useEffect(() => { api.fetchEvents().then(setEvents).catch(console.error); }, []);
  return (
    <div className="gap-24">
      <h2>Security Alerts & Audit Logs</h2>
      <div className="card">
        <div className="table-container">
          <table>
            <thead><tr><th>Time</th><th>Severity</th><th>Component</th><th>Event</th></tr></thead>
            <tbody>
              {events.map(e => (
                <tr key={e.id}>
                  <td>{e.timestamp}</td>
                  <td><Badge type={e.severity === 'INFO' ? 'info' : 'danger'}>{e.severity}</Badge></td>
                  <td>{e.component}</td>
                  <td>{e.event}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

// ----------------- MAIN APP ----------------- //

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');

  const renderContent = () => {
    switch (activeTab) {
      case 'overview': return <OverviewPage />;
      case 'security': return <SecurityDemoPage />;
      case 'fl': return <FederatedLearningPage />;
      case 'validation': return <ValidationPage />;

      case 'clients': return <ClientsPage />;
      case 'alerts': return <AlertsPage />;
      default: return <div>Not implemented</div>;
    }
  };

  return (
    <div className="app-container">
      <div className="sidebar">
        <div className="sidebar-header">
          <Shield size={20} style={{ marginRight: 10, color: 'var(--color-info)' }} />
          TrustGuard-FL
        </div>
        <div className="sidebar-nav">
          <button className={`nav-item ${activeTab === 'overview' ? 'active' : ''}`} onClick={() => setActiveTab('overview')}>
            <LayoutDashboard size={18} /> Overview
          </button>
          <button className={`nav-item ${activeTab === 'fl' ? 'active' : ''}`} onClick={() => setActiveTab('fl')}>
            <Activity size={18} /> Federated Learning
          </button>
          <button className={`nav-item ${activeTab === 'security' ? 'active' : ''}`} onClick={() => setActiveTab('security')}>
            <Lock size={18} /> Security & Demo
          </button>
          <button className={`nav-item ${activeTab === 'clients' ? 'active' : ''}`} onClick={() => setActiveTab('clients')}>
            <Database size={18} /> Clients
          </button>
          <button className={`nav-item ${activeTab === 'alerts' ? 'active' : ''}`} onClick={() => setActiveTab('alerts')}>
            <AlertTriangle size={18} /> Alerts & Logs
          </button>
          <button className={`nav-item ${activeTab === 'validation' ? 'active' : ''}`} onClick={() => setActiveTab('validation')}>
            <Upload size={18} /> CSV Validation
          </button>

        </div>
      </div>
      <div className="main-content">
        <div className="topbar">
          <div style={{ fontWeight: 600 }}>System: <span className="text-success">● Operational</span></div>
          <div className="flex-row" style={{ gap: 16 }}>
            <Badge type="info">v1.0</Badge>
          </div>
        </div>
        <div className="content-area">
          {renderContent()}
        </div>
      </div>
    </div>
  );
}
