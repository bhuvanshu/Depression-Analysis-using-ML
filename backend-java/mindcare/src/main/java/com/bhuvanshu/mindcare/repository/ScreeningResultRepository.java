package com.bhuvanshu.mindcare.repository;

import com.bhuvanshu.mindcare.entity.ScreeningResult;
import org.springframework.data.jpa.repository.JpaRepository;

public interface ScreeningResultRepository
        extends JpaRepository<ScreeningResult, Long> {
}