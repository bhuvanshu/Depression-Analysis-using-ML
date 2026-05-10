import { useState, useEffect, useMemo } from 'react';
import {
  Users, Bell, TrendingUp, TrendingDown, UserCheck, AlertTriangle,
  Activity, Search, Download
} from 'lucide-react';
import {
  Chart as ChartJS,
  ArcElement, CategoryScale, LinearScale, BarElement, PointElement, LineElement,
  Tooltip, Legend, Filler
} from 'chart.js';
import { Doughnut, Bar } from 'react-chartjs-2';
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
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(true);
  
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

  const filteredData = useMemo(() => {
    if (!searchQuery.trim()) return students;
    const q = searchQuery.toLowerCase();
    return students.filter(s =>
      s.studentName?.toLowerCase().includes(q) ||
      s.enrollmentId?.toLowerCase().includes(q) ||
      s.department?.toLowerCase().includes(q)
    );
  }, [searchQuery, students]);

  // Risk distribution chart
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

  // Dynamically calculate department stats from student list
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

  const getInitials = (name) => name ? name.split(' ').map(n => n[0]).join('').toUpperCase() : '?';
  const getScoreClass = (level) => level?.toLowerCase() || 'low';

  // Calculate average score dynamically
  const avgScore = students.length > 0 
    ? students.reduce((acc, curr) => acc + (curr.probabilityScore || 0), 0) / students.length 
    : 0;

  return (
    <>
      <div className="dashboard-topbar">
        <div className="dashboard-greeting">
          <h1>Dashboard</h1>
          <p>Welcome back, {admin.name || 'Admin'}. Here's what's happening.</p>
        </div>
        <div className="dashboard-topbar-actions">
          <Button variant="secondary" size="sm" icon={Bell}>Alerts</Button>
          <Button variant="primary" size="sm" icon={Download}>Export</Button>
        </div>
      </div>

      {loading ? (
        <div style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
          Loading dashboard data from server...
        </div>
      ) : (
        <>
          <div className="stats-grid">
            <Card elevated className="stat-card animate-fade-in">
              <div className="stat-icon stat-icon-primary">
                <Users size={22} />
              </div>
              <div className="stat-info">
                <div className="stat-label">Total Students</div>
                <div className="stat-value">{summary.totalStudents}</div>
                <div className="stat-change stat-change-up">
                  <TrendingUp size={12} /> Live DB Count
                </div>
              </div>
            </Card>

            <Card elevated className="stat-card animate-fade-in delay-1">
              <div className="stat-icon stat-icon-success">
                <UserCheck size={22} />
              </div>
              <div className="stat-info">
                <div className="stat-label">Screenings Done</div>
                <div className="stat-value">{students.length}</div>
                <div className="stat-change stat-change-up">
                  <TrendingUp size={12} /> Total Records
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
                  <TrendingDown size={12} /> Needs Attention
                </div>
              </div>
            </Card>

            <Card elevated className="stat-card animate-fade-in delay-3">
              <div className="stat-icon stat-icon-warning">
                <Activity size={22} />
              </div>
              <div className="stat-info">
                <div className="stat-label">Avg Risk Score</div>
                <div className="stat-value">{(avgScore * 100).toFixed(0)}%</div>
                <div className="stat-change stat-change-up">
                  <Activity size={12} /> Institutional Average
                </div>
              </div>
            </Card>
          </div>

          <div className="charts-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))' }}>
            <Card elevated className="chart-card animate-fade-in-up delay-2">
              <div className="chart-header">
                <div>
                  <div className="chart-title">Risk Distribution</div>
                  <div className="chart-subtitle">Current screened population</div>
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
          </div>

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
                  </tr>
                </thead>
                <tbody>
                  {filteredData.map((row, i) => (
                    <tr key={row.enrollmentId || i}>
                      <td>
                        <div className="table-student-name">
                          <div className="table-avatar">{getInitials(row.studentName)}</div>
                          <span className="table-name-text">{row.studentName}</span>
                        </div>
                      </td>
                      <td><span className="table-enrollment">{row.enrollmentId}</span></td>
                      <td>{row.department}</td>
                      <td><RiskBadge level={row.riskLevel} /></td>
                      <td>
                        <span className={`table-score score-${getScoreClass(row.riskLevel)}`}>
                          {((row.probabilityScore || 0) * 100).toFixed(0)}%
                        </span>
                      </td>
                    </tr>
                  ))}
                  {filteredData.length === 0 && (
                    <tr>
                      <td colSpan="5" style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>
                        {students.length === 0 ? "No screenings available in the database yet." : `No students found matching "${searchQuery}"`}
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
              <div className="table-pagination">
                <span className="table-pagination-info">
                  Showing {filteredData.length} of {students.length} entries
                </span>
                <div className="table-pagination-controls">
                  <Button variant="ghost" size="sm" disabled>Previous</Button>
                  <Button variant="ghost" size="sm" disabled>Next</Button>
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </>
  );
}
