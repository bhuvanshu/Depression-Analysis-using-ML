import { useState, useEffect, useMemo } from 'react';
import { Search, Filter, ChevronDown, Download, User, MoreVertical, UserPlus } from 'lucide-react';
import Card from '../../components/common/Card';
import Button from '../../components/common/Button';
import RiskBadge from '../../components/common/RiskBadge';
import AddStudentModal from './AddStudentModal';
import { getDashboardStudents, getHighRiskStudents } from '../../services/api';
import './StudentsPage.css';

export default function StudentsPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [riskFilter, setRiskFilter] = useState('All');
  const [students, setStudents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const fetchStudents = async () => {
    try {
      setLoading(true);
      let data;
      if (riskFilter === 'High') {
        data = await getHighRiskStudents();
      } else {
        data = await getDashboardStudents();
        if (riskFilter !== 'All') {
          data = data.filter(s => s.riskLevel === riskFilter);
        }
      }
      setStudents(data || []);
    } catch (err) {
      console.error("Failed to fetch students", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStudents();
  }, [riskFilter]);

  const filteredData = useMemo(() => {
    if (!searchQuery.trim()) return students;
    const q = searchQuery.toLowerCase();
    return students.filter(s =>
      s.studentName?.toLowerCase().includes(q) ||
      s.enrollmentId?.toLowerCase().includes(q) ||
      s.department?.toLowerCase().includes(q)
    );
  }, [searchQuery, students]);

  const getInitials = (name) => name ? name.split(' ').map(n => n[0]).join('').toUpperCase() : '?';
  const getScoreClass = (level) => level?.toLowerCase() || 'low';

  return (
    <div className="students-page animate-fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-title">Student Records</h1>
          <p className="page-subtitle">Manage and monitor student screenings</p>
        </div>
        <div className="page-actions">
          <Button variant="secondary" icon={Download}>Export List</Button>
          <Button variant="primary" icon={UserPlus} onClick={() => setIsModalOpen(true)}>
            Add Student
          </Button>
        </div>
      </div>

      <AddStudentModal 
        isOpen={isModalOpen} 
        onClose={() => setIsModalOpen(false)} 
        onSuccess={fetchStudents}
      />

      <Card elevated className="students-table-container">
        {/* Table Toolbar */}
        <div className="table-toolbar">
          <div className="toolbar-search">
            <div className="input-wrapper">
              <Search size={18} className="input-icon" />
              <input
                type="text"
                className="input-field has-icon"
                placeholder="Search name, ID, or department..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
          </div>
          <div className="toolbar-filters">
            <div className="filter-group">
              <span className="filter-label">Risk Level:</span>
              <select 
                className="input-field" 
                value={riskFilter}
                onChange={(e) => setRiskFilter(e.target.value)}
                style={{ width: '140px' }}
              >
                <option value="All">All Levels</option>
                <option value="High">High Risk</option>
                <option value="Moderate">Moderate Risk</option>
                <option value="Low">Low Risk</option>
              </select>
            </div>
          </div>
        </div>

        {/* Data Table */}
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
                  <th>Score</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredData.map((row, i) => (
                  <tr key={row.enrollmentId || i}>
                    <td>
                      <div className="table-student-name">
                        <div className="table-avatar">{getInitials(row.studentName)}</div>
                        <div>
                          <span className="table-name-text">{row.studentName}</span>
                        </div>
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
                    <td>
                      <Button variant="ghost" size="sm" icon={User}>View</Button>
                    </td>
                  </tr>
                ))}
                {filteredData.length === 0 && (
                  <tr>
                    <td colSpan="6" className="table-empty-state">
                      No students found matching your criteria
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          )}
        </div>

        {/* Pagination */}
        <div className="table-pagination">
          <span className="table-pagination-info">
            Showing {filteredData.length} of {students.length} entries
          </span>
          <div className="table-pagination-controls">
            <Button variant="ghost" size="sm" disabled>Previous</Button>
            <Button variant="ghost" size="sm" disabled>Next</Button>
          </div>
        </div>
      </Card>
    </div>
  );
}
