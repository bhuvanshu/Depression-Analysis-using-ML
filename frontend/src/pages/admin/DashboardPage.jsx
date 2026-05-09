import { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Brain, LayoutDashboard, Users, FileBarChart, Settings, LogOut,
  Search, Bell, TrendingUp, TrendingDown, UserCheck, AlertTriangle,
  Activity, Menu, X, Download
} from 'lucide-react';
import {
  Chart as ChartJS,
  ArcElement, CategoryScale, LinearScale, BarElement, PointElement, LineElement,
  Tooltip, Legend, Filler
} from 'chart.js';
import { Doughnut, Bar, Line } from 'react-chartjs-2';
import Card from '../../components/common/Card';
import Button from '../../components/common/Button';
import RiskBadge from '../../components/common/RiskBadge';
import {
  MOCK_STUDENTS, MOCK_RESULTS, MOCK_RESPONSES,
  DEPARTMENT_STATS, MONTHLY_TRENDS, DASHBOARD_STATS
} from '../../data/mockData';
import './DashboardPage.css';

ChartJS.register(
  ArcElement, CategoryScale, LinearScale, BarElement,
  PointElement, LineElement, Tooltip, Legend, Filler
);

const chartDefaults = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      labels: { color: '#94A3B8', font: { family: 'Inter', size: 12 }, padding: 16 }
    },
    tooltip: {
      backgroundColor: '#1E293B',
      titleColor: '#F1F5F9',
      bodyColor: '#94A3B8',
      borderColor: 'rgba(255,255,255,0.06)',
      borderWidth: 1,
      cornerRadius: 8,
      padding: 12
    }
  }
};

export default function DashboardPage() {
  const navigate = useNavigate();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [activeNav, setActiveNav] = useState('dashboard');

  const admin = JSON.parse(localStorage.getItem('admin_auth') || '{}');

  const handleLogout = () => {
    localStorage.removeItem('admin_auth');
    navigate('/admin/login');
  };

  // Merge student + result data for table
  const tableData = useMemo(() => {
    return MOCK_STUDENTS.map(student => {
      const response = MOCK_RESPONSES.find(r => r.student_id === student.student_id);
      const result = response ? MOCK_RESULTS.find(r => r.response_id === response.response_id) : null;
      return { ...student, response, result };
    }).filter(s => s.result);
  }, []);

  const filteredData = useMemo(() => {
    if (!searchQuery.trim()) return tableData;
    const q = searchQuery.toLowerCase();
    return tableData.filter(s =>
      s.name.toLowerCase().includes(q) ||
      s.enrollment_id.toLowerCase().includes(q) ||
      s.department.toLowerCase().includes(q)
    );
  }, [searchQuery, tableData]);

  // Risk distribution chart
  const riskCounts = useMemo(() => {
    const counts = { Low: 0, Moderate: 0, High: 0 };
    MOCK_RESULTS.forEach(r => { counts[r.risk_level]++; });
    return counts;
  }, []);

  const riskChartData = {
    labels: ['Low Risk', 'Moderate Risk', 'High Risk'],
    datasets: [{
      data: [riskCounts.Low, riskCounts.Moderate, riskCounts.High],
      backgroundColor: ['rgba(34,197,94,0.8)', 'rgba(245,158,11,0.8)', 'rgba(239,68,68,0.8)'],
      borderColor: ['#22C55E', '#F59E0B', '#EF4444'],
      borderWidth: 2,
      hoverOffset: 6
    }]
  };

  // Department chart
  const deptChartData = {
    labels: DEPARTMENT_STATS.map(d => d.department.length > 12 ? d.department.slice(0, 12) + '…' : d.department),
    datasets: [
      {
        label: 'High Risk',
        data: DEPARTMENT_STATS.map(d => d.high),
        backgroundColor: 'rgba(239, 68, 68, 0.7)',
        borderRadius: 4
      },
      {
        label: 'Moderate',
        data: DEPARTMENT_STATS.map(d => d.moderate),
        backgroundColor: 'rgba(245, 158, 11, 0.7)',
        borderRadius: 4
      },
      {
        label: 'Low Risk',
        data: DEPARTMENT_STATS.map(d => d.low),
        backgroundColor: 'rgba(34, 197, 94, 0.7)',
        borderRadius: 4
      }
    ]
  };

  // Trend chart
  const trendChartData = {
    labels: MONTHLY_TRENDS.map(t => t.month),
    datasets: [
      {
        label: 'Total Screenings',
        data: MONTHLY_TRENDS.map(t => t.screenings),
        borderColor: '#6366F1',
        backgroundColor: 'rgba(99, 102, 241, 0.08)',
        fill: true,
        tension: 0.4,
        pointRadius: 4,
        pointHoverRadius: 6
      },
      {
        label: 'High Risk Identified',
        data: MONTHLY_TRENDS.map(t => t.highRisk),
        borderColor: '#EF4444',
        backgroundColor: 'rgba(239, 68, 68, 0.05)',
        fill: true,
        tension: 0.4,
        pointRadius: 4,
        pointHoverRadius: 6
      }
    ]
  };

  const getInitials = (name) => name ? name.split(' ').map(n => n[0]).join('').toUpperCase() : '?';
  const getScoreClass = (level) => level?.toLowerCase() || 'low';

  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'students', label: 'Students', icon: Users },
    { id: 'reports', label: 'Reports', icon: FileBarChart },
    { id: 'settings', label: 'Settings', icon: Settings }
  ];

  return (
    <div className="dashboard-layout">
      {/* Mobile Toggle */}
      <button className="sidebar-toggle" onClick={() => setSidebarOpen(!sidebarOpen)}>
        {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
      </button>

      {/* Sidebar */}
      <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-header">
          <div className="sidebar-logo">
            <div className="sidebar-logo-icon">
              <Brain size={20} color="white" />
            </div>
            <div>
              <div className="sidebar-logo-text">MindScreen</div>
              <div className="sidebar-logo-sub">Admin Panel</div>
            </div>
          </div>
        </div>

        <nav className="sidebar-nav">
          {navItems.map(item => (
            <button
              key={item.id}
              className={`sidebar-nav-item ${activeNav === item.id ? 'active' : ''}`}
              onClick={() => setActiveNav(item.id)}
            >
              <item.icon size={18} />
              {item.label}
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="sidebar-user">
            <div className="sidebar-user-avatar">
              {getInitials(admin.name || 'Admin')}
            </div>
            <div className="sidebar-user-info">
              <div className="sidebar-user-name">{admin.name || 'Admin'}</div>
              <div className="sidebar-user-role">{admin.college || 'Institution'}</div>
            </div>
            <button
              onClick={handleLogout}
              style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', padding: 4 }}
              title="Logout"
            >
              <LogOut size={16} />
            </button>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="dashboard-main">
        {/* Top Bar */}
        <div className="dashboard-topbar">
          <div className="dashboard-greeting">
            <h1>Dashboard</h1>
            <p>Welcome back, {admin.name || 'Admin'}. Here's what's happening.</p>
          </div>
          <div className="dashboard-topbar-actions">
            <Button variant="secondary" size="sm" icon={Bell}>
              Alerts
            </Button>
            <Button variant="primary" size="sm" icon={Download}>
              Export
            </Button>
          </div>
        </div>

        {/* Stats Grid */}
        <div className="stats-grid">
          <Card elevated className="stat-card animate-fade-in">
            <div className="stat-icon stat-icon-primary">
              <Users size={22} />
            </div>
            <div className="stat-info">
              <div className="stat-label">Total Students</div>
              <div className="stat-value">{DASHBOARD_STATS.totalStudents}</div>
              <div className="stat-change stat-change-up">
                <TrendingUp size={12} /> +12 this month
              </div>
            </div>
          </Card>

          <Card elevated className="stat-card animate-fade-in delay-1">
            <div className="stat-icon stat-icon-success">
              <UserCheck size={22} />
            </div>
            <div className="stat-info">
              <div className="stat-label">Screenings Done</div>
              <div className="stat-value">{DASHBOARD_STATS.totalScreenings}</div>
              <div className="stat-change stat-change-up">
                <TrendingUp size={12} /> +28 this month
              </div>
            </div>
          </Card>

          <Card elevated className="stat-card animate-fade-in delay-2">
            <div className="stat-icon stat-icon-danger">
              <AlertTriangle size={22} />
            </div>
            <div className="stat-info">
              <div className="stat-label">High Risk</div>
              <div className="stat-value">{DASHBOARD_STATS.highRiskCount}</div>
              <div className="stat-change stat-change-down">
                <TrendingDown size={12} /> -3 vs last month
              </div>
            </div>
          </Card>

          <Card elevated className="stat-card animate-fade-in delay-3">
            <div className="stat-icon stat-icon-warning">
              <Activity size={22} />
            </div>
            <div className="stat-info">
              <div className="stat-label">Avg Risk Score</div>
              <div className="stat-value">{(DASHBOARD_STATS.avgScore * 100).toFixed(0)}%</div>
              <div className="stat-change stat-change-up">
                <TrendingDown size={12} /> -2% vs last month
              </div>
            </div>
          </Card>
        </div>

        {/* Charts */}
        <div className="charts-grid">
          <Card elevated className="chart-card animate-fade-in-up delay-2">
            <div className="chart-header">
              <div>
                <div className="chart-title">Risk Distribution</div>
                <div className="chart-subtitle">Current student population</div>
              </div>
            </div>
            <div className="chart-container chart-container-sm">
              <Doughnut
                data={riskChartData}
                options={{
                  ...chartDefaults,
                  cutout: '65%',
                  plugins: {
                    ...chartDefaults.plugins,
                    legend: { ...chartDefaults.plugins.legend, position: 'bottom' }
                  }
                }}
              />
            </div>
          </Card>

          <Card elevated className="chart-card animate-fade-in-up delay-3">
            <div className="chart-header">
              <div>
                <div className="chart-title">Department Breakdown</div>
                <div className="chart-subtitle">Risk levels by department</div>
              </div>
            </div>
            <div className="chart-container chart-container-sm">
              <Bar
                data={deptChartData}
                options={{
                  ...chartDefaults,
                  scales: {
                    x: {
                      stacked: true,
                      grid: { display: false },
                      ticks: { color: '#64748B', font: { size: 10 } }
                    },
                    y: {
                      stacked: true,
                      grid: { color: 'rgba(255,255,255,0.04)' },
                      ticks: { color: '#64748B' }
                    }
                  }
                }}
              />
            </div>
          </Card>

          <Card elevated className="chart-card chart-card-full animate-fade-in-up delay-4">
            <div className="chart-header">
              <div>
                <div className="chart-title">Screening Trends</div>
                <div className="chart-subtitle">Monthly screening activity & high-risk identification</div>
              </div>
            </div>
            <div className="chart-container">
              <Line
                data={trendChartData}
                options={{
                  ...chartDefaults,
                  scales: {
                    x: {
                      grid: { display: false },
                      ticks: { color: '#64748B' }
                    },
                    y: {
                      grid: { color: 'rgba(255,255,255,0.04)' },
                      ticks: { color: '#64748B' }
                    }
                  }
                }}
              />
            </div>
          </Card>
        </div>

        {/* Student Table */}
        <div className="table-section">
          <div className="table-header">
            <h2 className="table-title">Recent Screenings</h2>
            <div className="table-search">
              <div className="input-wrapper">
                <Search size={18} className="input-icon" />
                <input
                  type="text"
                  className="input-field has-icon"
                  placeholder="Search students..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
              </div>
            </div>
          </div>

          <div className="data-table-wrapper">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Student</th>
                  <th>Enrollment ID</th>
                  <th>Department</th>
                  <th>Risk Level</th>
                  <th>Score</th>
                  <th>Date</th>
                </tr>
              </thead>
              <tbody>
                {filteredData.map((row) => (
                  <tr key={row.student_id}>
                    <td>
                      <div className="table-student-name">
                        <div className="table-avatar">{getInitials(row.name)}</div>
                        <span className="table-name-text">{row.name}</span>
                      </div>
                    </td>
                    <td><span className="table-enrollment">{row.enrollment_id}</span></td>
                    <td>{row.department}</td>
                    <td><RiskBadge level={row.result?.risk_level} /></td>
                    <td>
                      <span className={`table-score score-${getScoreClass(row.result?.risk_level)}`}>
                        {((row.result?.probability_score || 0) * 100).toFixed(0)}%
                      </span>
                    </td>
                    <td>{row.response?.submitted_at ? new Date(row.response.submitted_at).toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' }) : '—'}</td>
                  </tr>
                ))}
                {filteredData.length === 0 && (
                  <tr>
                    <td colSpan="6" style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>
                      No students found matching "{searchQuery}"
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
            <div className="table-pagination">
              <span className="table-pagination-info">
                Showing {filteredData.length} of {tableData.length} entries
              </span>
              <div className="table-pagination-controls">
                <Button variant="ghost" size="sm" disabled>Previous</Button>
                <Button variant="ghost" size="sm" disabled>Next</Button>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
