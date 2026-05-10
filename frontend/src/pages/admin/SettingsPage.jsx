import Card from '../../components/common/Card';

export default function SettingsPage() {
  return (
    <div className="animate-fade-in">
      <div className="page-header" style={{ marginBottom: 'var(--space-6)' }}>
        <h1 className="page-title" style={{ fontSize: 'var(--font-size-2xl)', fontWeight: 700 }}>Settings</h1>
        <p className="page-subtitle" style={{ color: 'var(--text-secondary)' }}>Configure system preferences</p>
      </div>
      <Card elevated>
        <p style={{ color: 'var(--text-muted)', padding: 'var(--space-8)', textAlign: 'center' }}>
          Settings module coming soon.
        </p>
      </Card>
    </div>
  );
}
