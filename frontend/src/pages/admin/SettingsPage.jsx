import { useState, useEffect } from 'react';
import { Database, ShieldCheck, Cpu, Clock, HardDrive, Mail, Building, Info } from 'lucide-react';
import Card from '../../components/common/Card';
import './SettingsPage.css';

export default function SettingsPage() {
  const admin = JSON.parse(localStorage.getItem('admin_auth') || '{}');
  const collegeName = admin.college || 'Institutional Partner';

  const [systemStats, setSystemStats] = useState([
    { label: 'ML API Connection', status: 'Checking…', icon: Cpu, color: 'var(--text-muted)' },
    { label: 'Database Status', status: 'Checking…', icon: Database, color: 'var(--text-muted)' },
    { label: 'System Security', status: 'Active', icon: ShieldCheck, color: 'var(--accent-primary)' },
  ]);

  useEffect(() => {
    const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8080/api';

    // Check ML API
    fetch(`${BASE_URL}/dashboard/summary`, { method: 'GET' })
      .then(res => {
        setSystemStats(prev => prev.map(s =>
          s.label === 'ML API Connection'
            ? { ...s, status: res.ok ? 'Connected' : 'Error', color: res.ok ? 'var(--accent-success)' : 'var(--accent-danger)' }
            : s
        ));
        // If the API responds, DB is reachable through it
        setSystemStats(prev => prev.map(s =>
          s.label === 'Database Status'
            ? { ...s, status: res.ok ? 'Healthy' : 'Unreachable', color: res.ok ? 'var(--accent-success)' : 'var(--accent-danger)' }
            : s
        ));
      })
      .catch(() => {
        setSystemStats(prev => prev.map(s =>
          s.label === 'ML API Connection' || s.label === 'Database Status'
            ? { ...s, status: 'Offline', color: 'var(--accent-danger)' }
            : s
        ));
      });
  }, []);

  return (
    <div className="settings-page animate-fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-title">System Settings</h1>
          <p className="page-subtitle">Admin configuration & System status</p>
        </div>
      </div>

      <div className="settings-grid">
        <Card elevated className="settings-card">
          <h3 className="settings-section-title">Admin Profile</h3>
          <div className="profile-details">
            <div className="detail-row">
              <Building size={16} style={{ color: 'var(--text-muted)', marginTop: '4px' }} />
              <div>
                <label>Institution Name</label>
                <p>{collegeName}</p>
              </div>
            </div>
            <div className="detail-row">
              <Mail size={16} style={{ color: 'var(--text-muted)', marginTop: '4px' }} />
              <div>
                <label>Admin Email</label>
                <p>{admin.email || 'admin@university.edu'}</p>
              </div>
            </div>
          </div>
        </Card>

        <Card elevated className="settings-card">
          <h3 className="settings-section-title">System & ML Status</h3>
          <div className="status-list">
            {systemStats.map((stat, i) => (
              <div key={i} className="status-item">
                <div className="status-info">
                  <stat.icon size={18} style={{ color: stat.color }} />
                  <span>{stat.label}</span>
                </div>
                <div className="status-badge" style={{ color: stat.color }}>
                  {stat.status === 'Connected' || stat.status === 'Healthy' || stat.status === 'Active' ? 'Live ✅' : `${stat.status} ⚠️`}
                </div>
              </div>
            ))}
          </div>
        </Card>

        <Card elevated className="settings-card deployment-card">
          <h3 className="settings-section-title">Deployment Info</h3>
          <div className="deployment-grid">
            <div className="deploy-item">
              <Clock size={16} />
              <span>Last Sync: 2 mins ago</span>
            </div>
            <div className="deploy-item">
              <HardDrive size={16} />
              <span>Version: v1.0.4-stable</span>
            </div>
          </div>
        </Card>

        <Card elevated className="placeholder-card">
          <div className="placeholder-content">
            <Info size={24} />
            <p>Future configurations like questionnaire thresholds and notification settings will appear here.</p>
          </div>
        </Card>
      </div>
    </div>
  );
}
