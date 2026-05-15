package com.bhuvanshu.mindcare.repository;

import com.bhuvanshu.mindcare.entity.College;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface CollegeRepository
        extends JpaRepository<College, Long> {

    boolean existsByAdminEmail(String adminEmail);

    Optional<College> findByCollegeName(String collegeName);
}