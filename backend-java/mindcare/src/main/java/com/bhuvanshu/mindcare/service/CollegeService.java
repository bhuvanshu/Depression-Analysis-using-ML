package com.bhuvanshu.mindcare.service;

import com.bhuvanshu.mindcare.entity.College;
import com.bhuvanshu.mindcare.repository.CollegeRepository;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class CollegeService {

    private static final Logger logger = LoggerFactory.getLogger(CollegeService.class);

    @Autowired
    private CollegeRepository collegeRepository;

    /**
     * Resolves a College entity from the collegeName.
     * Returns null if collegeName is blank (graceful fallback for backward compatibility).
     */
    public College resolveCollege(String collegeName) {
        if (collegeName == null || collegeName.trim().isEmpty()) {
            return null;
        }
        List<College> colleges = collegeRepository.findByCollegeName(collegeName);
        College college = (colleges != null && !colleges.isEmpty()) ? colleges.get(0) : null;
        if (college == null) {
            logger.warn("College not found for name '{}', falling back to unfiltered data.", collegeName);
        }
        return college;
    }
}
