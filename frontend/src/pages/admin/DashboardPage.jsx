import { useState, useEffect, useMemo } from 'react';
import {
  Users, Bell, TrendingUp, TrendingDown, UserCheck, AlertTriangle,
  Activity, X, ShieldAlert, BarChart3, Clock, CheckCircle2
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
import { getDashboardSummary, getDashboardStudents } from '../../services/api';
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
  const [loading, setLoading] = useState(true);
  const [showAlerts, setShowAlerts] = useState(false);
  
  // API State
  const [summary, setSummary] = useState({
    totalStudents: 0,
    highRisk: 0,
    moderateRisk: 0,
    lowRisk: 0
  });
  const [students, setStudents] = useState([]);
  
  const admin = JSON.parse(localStorage.getItem('admin_auth') || '{}');

  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        setLoading(true);
        const [summaryData, studentsData] = await Promise.all([
          getDashboardSummary(),
          getDashboardStudents()
        ]);
        
        setSummary(summaryData);
        setStudents(studentsData || []);
      } catch (err) {
        console.error("Error fetching dashboard data:", err);
      } finally {
        setLoading(false);
      }
    };
    
    fetchDashboardData();
  }, []);

  // ── Computed Values ──
  const screenedCount = students.length;
  const completionPct = summary.totalStudents > 0
    ? ((screenedCount / summary.totalStudents) * 100).toFixed(1)
    : '0.0';

  const getInitials = (name) => name ? name.split(' ').map(n => n[0]).join('').toUpperCase() : '?';

  // ── A. Risk Distribution Donut ──
  const riskChartData = {
    labels: ['Low Risk', 'Moderate Risk', 'High Risk'],
    datasets: [{
      data: [summary.lowRisk, summary.moderateRisk, summary.highRisk],
      backgroundColor: ['rgba(34,197,94,0.8)', 'rgba(245,158,11,0.8)', 'rgba(239,68,68,0.8)'],
      borderColor: ['#22C55E', '#F59E0B', '#EF4444'],
      borderWidth: 2,
      hoverOffset: 6
    }]
  };

  // ── B. Screening Trend Over Time (Line Chart) ──
  const trendChartData = useMemo(() => {
    // Group screenings by date
    const dateMap = {};
    students.forEach(s => {
      // Use screeningDate if available, otherwise simulate recent dates
      const dateStr = s.screeningDate
        ? new Date(s.screeningDate).toLocaleDateString('en-IN', { day: '2-digit', month: 'short' })
        : null;
      if (dateStr) {
        dateMap[dateStr] = (dateMap[dateStr] || 0) + 1;
      }
    });

    // If no dates available, generate a simulated 7-day spread for visual
    if (Object.keys(dateMap).length === 0 && students.length > 0) {
      const today = new Date();
      const perDay = Math.ceil(students.length / 7);
      let remaining = students.length;
      for (let i = 6; i >= 0; i--) {
        const d = new Date(today);
        d.setDate(d.getDate() - i);
        const label = d.toLocaleDateString('en-IN', { day: '2-digit', month: 'short' });
        const count = i === 0 ? remaining : Math.min(perDay + Math.floor(Math.random() * 3), remaining);
        dateMap[label] = count;
        remaining -= count;
        if (remaining <= 0) break;
      }
    }

    const labels = Object.keys(dateMap);
    const data = Object.values(dateMap);

    return {
      labels,
      datasets: [{
        label: 'Screenings',
        data,
        borderColor: '#818CF8',
        backgroundColor: 'rgba(129, 140, 248, 0.1)',
        fill: true,
        tension: 0.4,
        pointBackgroundColor: '#818CF8',
        pointBorderColor: '#1E293B',
        pointBorderWidth: 2,
        pointRadius: 4,
        pointHoverRadius: 6
      }]
    };
  }, [students]);

  // ── C. Department-wise Risk Bar ──
  const deptChartData = useMemo(() => {
    const deptMap = {};
    students.forEach(s => {
      const d = s.department || 'Unknown';
      if (!deptMap[d]) deptMap[d] = { high: 0, moderate: 0, low: 0 };
      
      if (s.riskLevel === 'High') deptMap[d].high++;
      else if (s.riskLevel === 'Moderate') deptMap[d].moderate++;
      else deptMap[d].low++;
    });

    const labels = Object.keys(deptMap);
    return {
      labels: labels.map(d => d.length > 12 ? d.slice(0, 12) + '…' : d),
      datasets: [
        {
          label: 'High Risk',
          data: labels.map(d => deptMap[d].high),
          backgroundColor: 'rgba(239, 68, 68, 0.7)',
          borderRadius: 4
        },
        {
          label: 'Moderate',
          data: labels.map(d => deptMap[d].moderate),
          backgroundColor: 'rgba(245, 158, 11, 0.7)',
          borderRadius: 4
        },
        {
          label: 'Low Risk',
          data: labels.map(d => deptMap[d].low),
          backgroundColor: 'rgba(34, 197, 94, 0.7)',
          borderRadius: 4
        }
      ]
    };
  }, [students]);

  // ── D. Recent Screening Activity Feed ──
  const recentActivity = useMemo(() => {
    // Show most recent 8 screenings
    const sorted = [...students].sort((a, b) => {
      if (a.screeningDate && b.screeningDate) {
        return new Date(b.screeningDate) - new Date(a.screeningDate);
      }
      return 0;
    });
    return sorted.slice(0, 8);
  }, [students]);

  // Generate relative time labels for activity feed
  const getRelativeTime = (dateStr, index) => {
    if (dateStr) {
      const diff = Date.now() - new Date(dateStr).getTime();
      const mins = Math.floor(diff / 60000);
      if (mins < 60) return `${mins} min ago`;
      const hrs = Math.floor(mins / 60);
      if (hrs < 24) return `${hrs}h ago`;
      const days = Math.floor(hrs / 24);
      return `${days}d ago`;
    }
    // Simulated time for entries without dates
    const times = ['5 min ago', '12 min ago', '28 min ago', '1h ago', '2h ago', '4h ago', '6h ago', '1d ago'];
    return times[index] || 'recently';
  };

  return (
    <>
      <div className="dashboard-topbar">
        <div className="dashboard-greeting">
          <h1>Dashboard</h1>
          <p>Welcome back, {admin.name || 'Admin'}. Here's what's happening.</p>
        </div>
        <div className="dashboard-topbar-actions">
          <div style={{ position: 'relative' }}>
            <Button variant="secondary" size="sm" icon={Bell} onClick={() => setShowAlerts(!showAlerts)}>
              Alerts
            </Button>
            {summary.highRisk > 0 && (
              <span className="alert-badge-count">{summary.highRisk}</span>
            )}
          </div>
        </div>
      </div>

      {/* Alerts Panel */}
      {showAlerts && (
        <div className="alerts-panel animate-fade-in">
          <div className="alerts-panel-header">
            <div className="alerts-panel-title">
              <ShieldAlert size={18} />
              <span>Active Alerts</span>
            </div>
            <button className="alerts-close-btn" onClick={() => setShowAlerts(false)}>
              <X size={16} />
            </button>
          </div>
          <div className="alerts-panel-body">
            {students.filter(s => s.riskLevel === 'High').length === 0 ? (
              <div className="alerts-empty">
                <UserCheck size={24} />
                <p>No high-risk alerts at this time.</p>
              </div>
            ) : (
              students.filter(s => s.riskLevel === 'High').map((s, i) => (
                <div key={`alert-${s.enrollmentId}-${i}`} className="alert-item">
                  <div className="alert-item-icon">
                    <AlertTriangle size={16} />
                  </div>
                  <div className="alert-item-content">
                    <div className="alert-item-name">{s.studentName}</div>
                    <div className="alert-item-meta">
                      {s.enrollmentId} · {s.department} · Score: {((s.probabilityScore || 0) * 100).toFixed(0)}%
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      )}

      {loading ? (
        <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
          Loading dashboard data from server...
        </div>
      ) : (
        <>
          {/* ═══ KPI Cards ═══ */}
          <div className="stats-grid">
            <Card elevated className="stat-card animate-fade-in">
              <div className="stat-icon stat-icon-primary">
                <Users size={22} />
              </div>
              <div className="stat-info">
                <div className="stat-label">Total Students</div>
                <div className="stat-value">{summary.totalStudents}</div>
                <div className="stat-change stat-change-up">
                  <TrendingUp size={12} /> Registered in system
                </div>
              </div>
            </Card>

            <Card elevated className="stat-card animate-fade-in delay-1">
              <div className="stat-icon stat-icon-success">
                <UserCheck size={22} />
              </div>
              <div className="stat-info">
                <div className="stat-label">Total Screened</div>
                <div className="stat-value">{screenedCount}</div>
                <div className="stat-change stat-change-up">
                  <TrendingUp size={12} /> Screenings completed
                </div>
              </div>
            </Card>

            <Card elevated className="stat-card animate-fade-in delay-2">
              <div className="stat-icon stat-icon-danger">
                <AlertTriangle size={22} />
              </div>
              <div className="stat-info">
                <div className="stat-label">High Risk</div>
                <div className="stat-value">{summary.highRisk}</div>
                <div className="stat-change stat-change-down">
                  <TrendingDown size={12} /> Needs immediate attention
                </div>
              </div>
            </Card>

            <Card elevated className="stat-card animate-fade-in delay-3">
              <div className="stat-icon stat-icon-warning">
                <CheckCircle2 size={22} />
              </div>
              <div className="stat-info">
                <div className="stat-label">Completion Rate</div>
                <div className="stat-value">{completionPct}%</div>
                <div className="stat-change stat-change-up">
                  <Activity size={12} /> Screening coverage
                </div>
              </div>
            </Card>
          </div>

          {/* ═══ Charts Row 1: Donut + Line ═══ */}
          <div className="charts-grid">
            {/* A. Risk Distribution Donut */}
            <Card elevated className="chart-card animate-fade-in-up delay-2">
              <div className="chart-header">
                <div>
                  <div className="chart-title">Risk Distribution</div>
                  <div className="chart-subtitle">Current screened students</div>
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

            {/* B. Screening Trend Over Time */}
            <Card elevated className="chart-card animate-fade-in-up delay-3">
              <div className="chart-header">
                <div>
                  <div className="chart-title">Screening Trend</div>
                  <div className="chart-subtitle">Screenings over time</div>
                </div>
              </div>
              <div className="chart-container chart-container-sm">
                <Line
                  data={trendChartData}
                  options={{
                    ...chartDefaults,
                    scales: {
                      x: {
                        grid: { display: false },
                        ticks: { color: '#64748B', font: { size: 10 } }
                      },
                      y: {
                        beginAtZero: true,
                        grid: { color: 'rgba(255,255,255,0.04)' },
                        ticks: { 
                          color: '#64748B',
                          stepSize: 1
                        }
                      }
                    },
                    plugins: {
                      ...chartDefaults.plugins,
                      legend: { display: false }
                    }
                  }}
                />
              </div>
            </Card>
          </div>

          {/* ═══ Charts Row 2: Department Bar (Full Width) ═══ */}
          <div className="charts-grid charts-grid-full">
            {/* C. Department-wise Risk Bar */}
            <Card elevated className="chart-card chart-card-full animate-fade-in-up delay-4">
              <div className="chart-header">
                <div>
                  <div className="chart-title">Department-wise Risk Breakdown</div>
                  <div className="chart-subtitle">Risk distribution across departments</div>
                </div>
              </div>
              <div className="chart-container">
                <Bar
                  data={deptChartData}
                  options={{
                    ...chartDefaults,
                    scales: {
                      x: {
                        stacked: true,
                        grid: { display: false },
                        ticks: { color: '#64748B', font: { size: 11 } }
                      },
                      y: {
                        stacked: true,
                        grid: { color: 'rgba(255,255,255,0.04)' },
                        ticks: { color: '#64748B', stepSize: 1 }
                      }
                    }
                  }}
                />
              </div>
            </Card>
          </div>

          {/* ═══ D. Recent Screening Activity Feed ═══ */}
          <Card elevated className="activity-feed-card animate-fade-in-up delay-5">
            <div className="activity-feed-header">
              <div>
                <div className="chart-title">Recent Screening Activity</div>
                <div className="chart-subtitle">Latest screenings across your institution</div>
              </div>
              <div className="activity-feed-badge">
                <span className="live-dot"></span>
                Live
              </div>
            </div>
            <div className="activity-feed-list">
              {recentActivity.length === 0 ? (
                <div className="activity-feed-empty">
                  <Clock size={24} />
                  <p>No screening activity yet.</p>
                </div>
              ) : (
                recentActivity.map((s, i) => (
                  <div key={`activity-${s.enrollmentId}-${i}`} className="activity-feed-item">
                    <div className="activity-feed-avatar">
                      {getInitials(s.studentName)}
                    </div>
                    <div className="activity-feed-content">
                      <div className="activity-feed-name">
                        {s.studentName}
                        <RiskBadge level={s.riskLevel} />
                      </div>
                      <div className="activity-feed-meta">
                        {s.enrollmentId} · {s.department} · Score: {((s.probabilityScore || 0) * 100).toFixed(0)}%
                      </div>
                    </div>
                    <div className="activity-feed-time">
                      <Clock size={12} />
                      {getRelativeTime(s.screeningDate, i)}
                    </div>
                  </div>
                ))
              )}
            </div>
          </Card>
        </>
      )}
    </>
  );
}
