import { useState, useEffect, useMemo } from 'react';
import {
  Search, UserPlus, AlertTriangle, Eye, Upload,
  Filter, ChevronDown, Shield
} from 'lucide-react';
import Card from '../../components/common/Card';
import Button from '../../components/common/Button';
import RiskBadge from '../../components/common/RiskBadge';
import AddStudentModal from './AddStudentModal';
import StudentDetailModal from './StudentDetailModal';
import { getDashboardStudents, getHighRiskStudents } from '../../services/api';
import './StudentsPage.css';

export default function StudentsPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [riskFilter, setRiskFilter] = useState('All');
  const [deptFilter, setDeptFilter] = useState('All');
  const [students, setStudents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [selectedStudent, setSelectedStudent] = useState(null);

  const fetchStudents = async () => {
    try {
      setLoading(true);
      const data = await getDashboardStudents();
      setStudents(data || []);
    } catch (err) {
      console.error("Failed to fetch students", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStudents();
  }, []);

  // ── Extract unique departments for filter dropdown ──
  const departments = useMemo(() => {
    const depts = new Set(students.map(s => s.department).filter(Boolean));
    return ['All', ...Array.from(depts).sort()];
  }, [students]);

  // ── High risk students pinned at top ──
  const highRiskStudents = useMemo(() => {
    return students.filter(s => s.riskLevel === 'High');
  }, [students]);

  // ── Filtered + searched data ──
  const filteredData = useMemo(() => {
    let data = students;

    // Risk filter
    if (riskFilter !== 'All') {
      data = data.filter(s => s.riskLevel === riskFilter);
    }

    // Department filter
    if (deptFilter !== 'All') {
      data = data.filter(s => s.department === deptFilter);
    }

    // Search
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      data = data.filter(s =>
        s.studentName?.toLowerCase().includes(q) ||
        s.enrollmentId?.toLowerCase().includes(q) ||
        s.department?.toLowerCase().includes(q)
      );
    }

    // Sort: High risk first, then Moderate, then Low
    const riskOrder = { High: 0, Moderate: 1, Low: 2 };
    data.sort((a, b) => (riskOrder[a.riskLevel] ?? 3) - (riskOrder[b.riskLevel] ?? 3));

    return data;
  }, [searchQuery, riskFilter, deptFilter, students]);

  const getInitials = (name) => name ? name.split(' ').map(n => n[0]).join('').toUpperCase() : '?';
  const getScoreClass = (level) => level?.toLowerCase() || 'low';

  return (
    <div className="students-page animate-fade-in">
      {/* ═══ Page Header ═══ */}
      <div className="page-header">
        <div>
          <h1 className="page-title">Student Records</h1>
          <p className="page-subtitle">Operational management layer — manage, monitor & intervene</p>
        </div>
        <div className="page-actions">
          <Button variant="primary" icon={UserPlus} onClick={() => setIsAddModalOpen(true)}>
            Add Student
          </Button>
        </div>
      </div>

      {/* ═══ High Risk Priority Section ═══ */}
      {highRiskStudents.length > 0 && (
        <Card elevated className="high-risk-section animate-fade-in">
          <div className="high-risk-header">
            <div className="high-risk-title">
              <div className="high-risk-icon">
                <AlertTriangle size={18} />
              </div>
              <div>
                <h3>High Risk Students — Immediate Attention Required</h3>
                <p>{highRiskStudents.length} student{highRiskStudents.length > 1 ? 's' : ''} flagged for follow-up</p>
              </div>
            </div>
          </div>
          <div className="high-risk-grid">
            {highRiskStudents.map((s, i) => (
              <div
                key={`hr-${s.enrollmentId}-${i}`}
                className="high-risk-card"
                onClick={() => setSelectedStudent(s)}
              >
                <div className="high-risk-card-avatar">
                  {getInitials(s.studentName)}
                </div>
                <div className="high-risk-card-info">
                  <div className="high-risk-card-name">{s.studentName}</div>
                  <div className="high-risk-card-meta">{s.enrollmentId} · {s.department}</div>
                </div>
                <div className="high-risk-card-score">
                  <span className="score-high">
                    {((s.probabilityScore || 0) * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* ═══ Filters & Search Toolbar ═══ */}
      <Card elevated className="students-table-container">
        <div className="table-toolbar">
          <div className="toolbar-search">
            <div className="input-wrapper">
              <Search size={18} className="input-icon" />
              <input
                type="text"
                className="input-field has-icon"
                placeholder="Search by name or enrollment ID..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
          </div>
          <div className="toolbar-filters">
            <div className="filter-group">
              <Filter size={14} className="filter-icon" />
              <span className="filter-label">Risk:</span>
              <select
                className="input-field filter-select"
                value={riskFilter}
                onChange={(e) => setRiskFilter(e.target.value)}
              >
                <option value="All">All Levels</option>
                <option value="High">High Risk</option>
                <option value="Moderate">Moderate</option>
                <option value="Low">Low Risk</option>
              </select>
            </div>
            <div className="filter-group">
              <span className="filter-label">Dept:</span>
              <select
                className="input-field filter-select"
                value={deptFilter}
                onChange={(e) => setDeptFilter(e.target.value)}
              >
                {departments.map(d => (
                  <option key={d} value={d}>{d === 'All' ? 'All Departments' : d}</option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {/* ═══ Student Data Table ═══ */}
        <div className="data-table-wrapper">
          {loading ? (
            <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>
              Loading student records...
            </div>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  <th>Student</th>
                  <th>Enrollment ID</th>
                  <th>Department</th>
                  <th>Risk Level</th>
                  <th>Probability Score</th>
                  <th>Screening Date</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredData.map((row, i) => (
                  <tr
                    key={`${row.enrollmentId}-${i}`}
                    className={row.riskLevel === 'High' ? 'row-high-risk' : ''}
                    onClick={() => setSelectedStudent(row)}
                    style={{ cursor: 'pointer' }}
                  >
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
                        {((row.probabilityScore || 0) * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td className="table-date">
                      {row.screeningDate
                        ? new Date(row.screeningDate).toLocaleDateString('en-IN', {
                            day: '2-digit', month: 'short', year: 'numeric'
                          })
                        : '—'
                      }
                    </td>
                    <td>
                      <Button
                        variant="ghost"
                        size="sm"
                        icon={Eye}
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedStudent(row);
                        }}
                      >
                        View
                      </Button>
                    </td>
                  </tr>
                ))}
                {filteredData.length === 0 && (
                  <tr>
                    <td colSpan="7" className="table-empty-state">
                      No students found matching your criteria
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          )}
        </div>

        {/* ═══ Pagination ═══ */}
        <div className="table-pagination">
          <span className="table-pagination-info">
            Showing {filteredData.length} of {students.length} students
            {riskFilter !== 'All' && ` · Filtered by: ${riskFilter} Risk`}
            {deptFilter !== 'All' && ` · ${deptFilter}`}
          </span>
        </div>
      </Card>

      {/* ═══ Modals ═══ */}
      <AddStudentModal
        isOpen={isAddModalOpen}
        onClose={() => setIsAddModalOpen(false)}
        onSuccess={fetchStudents}
      />

      <StudentDetailModal
        isOpen={!!selectedStudent}
        student={selectedStudent}
        onClose={() => setSelectedStudent(null)}
      />
    </div>
  );
}
