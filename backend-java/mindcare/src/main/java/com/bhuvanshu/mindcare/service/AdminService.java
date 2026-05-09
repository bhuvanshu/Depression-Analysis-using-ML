package com.bhuvanshu.mindcare.service;

import com.bhuvanshu.mindcare.dto.AdminSignupRequest;
import com.bhuvanshu.mindcare.entity.Admin;
import com.bhuvanshu.mindcare.entity.College;
import com.bhuvanshu.mindcare.repository.AdminRepository;
import com.bhuvanshu.mindcare.repository.CollegeRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class AdminService {

    @Autowired
    private AdminRepository adminRepository;

    @Autowired
    private CollegeRepository collegeRepository;

    public String signup(AdminSignupRequest request) {

        boolean exists = collegeRepository
                .existsByAdminEmail(request.getAdminEmail());

        if (exists) {
            return "Admin email already exists";
        }

        College college = new College();
        college.setCollegeName(request.getCollegeName());
        college.setAdminEmail(request.getAdminEmail());
        college.setPasswordHash(request.getPassword());

        collegeRepository.save(college);

        Admin admin = new Admin();
        admin.setAdminName(request.getAdminName());
        admin.setAdminEmail(request.getAdminEmail());
        admin.setPasswordHash(request.getPassword());
        admin.setCollege(college);

        adminRepository.save(admin);

        return "Admin signup successful";
    }
}