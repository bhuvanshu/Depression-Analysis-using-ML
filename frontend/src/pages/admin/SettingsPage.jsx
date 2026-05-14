import { Database, ShieldCheck, Cpu, Clock, HardDrive, Mail, Building, Info } from 'lucide-react';
import Card from '../../components/common/Card';
import './SettingsPage.css';

export default function SettingsPage() {
  const admin = JSON.parse(localStorage.getItem('admin_auth') || '{}');
  const collegeName = localStorage.getItem('collegeName') || 'Institutional Partner';

  const systemStats = [
    { label: 'ML API Connection', status: 'Connected', icon: Cpu, color: 'var(--accent-success)' },
    { label: 'Database Status', status: 'Healthy', icon: Database, color: 'var(--accent-success)' },
    { label: 'System Security', status: 'Active', icon: ShieldCheck, color: 'var(--accent-primary)' },
  ];

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
                <div className="status-badge">Live ✅</div>
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
