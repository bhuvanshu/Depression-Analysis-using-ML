package com.bhuvanshu.mindcare.repository;

import com.bhuvanshu.mindcare.entity.College;
import com.bhuvanshu.mindcare.entity.Student;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

public interface StudentRepository
        extends JpaRepository<Student, Long> {

    Optional<Student> findByEnrollmentId(String enrollmentId);

    boolean existsByEnrollmentId(String enrollmentId);

    List<Student> findByCollege(College college);

    long countByCollege(College college);
}