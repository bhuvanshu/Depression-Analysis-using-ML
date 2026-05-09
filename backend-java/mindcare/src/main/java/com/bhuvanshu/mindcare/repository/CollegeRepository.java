package com.bhuvanshu.mindcare.repository;

import com.bhuvanshu.mindcare.entity.College;
import org.springframework.data.jpa.repository.JpaRepository;

public interface CollegeRepository
        extends JpaRepository<College, Long> {

    boolean existsByAdminEmail(String adminEmail);
}