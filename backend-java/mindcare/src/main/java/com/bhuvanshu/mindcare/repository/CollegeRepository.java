package com.bhuvanshu.mindcare.repository;

import com.bhuvanshu.mindcare.entity.College;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;
import java.util.Optional;

public interface CollegeRepository
        extends JpaRepository<College, Long> {

    boolean existsByAdminEmail(String adminEmail);

    List<College> findByCollegeName(String collegeName);

    Optional<College> findByAdminEmail(String adminEmail);
}