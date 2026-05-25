package com.bhuvanshu.mindcare.service;

import com.bhuvanshu.mindcare.dto.AdminLoginRequest;
import com.bhuvanshu.mindcare.dto.AdminLoginResponse;
import com.bhuvanshu.mindcare.dto.AdminSignupRequest;
import com.bhuvanshu.mindcare.entity.Admin;
import com.bhuvanshu.mindcare.entity.College;
import com.bhuvanshu.mindcare.repository.AdminRepository;
import com.bhuvanshu.mindcare.repository.CollegeRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import java.util.Optional;

@Service
public class AdminService {

    @Autowired
    private AdminRepository adminRepository;

    @Autowired
    private CollegeRepository collegeRepository;

    @Autowired
    private PasswordEncoder passwordEncoder;

    public String signup(AdminSignupRequest request) {

        boolean exists = collegeRepository
                .existsByAdminEmail(request.getAdminEmail());

        if (exists) {
            throw new IllegalArgumentException("Admin email already exists");
        }

        String encodedPassword = passwordEncoder.encode(request.getPassword());

        College college = new College();
        college.setCollegeName(request.getCollegeName());
        college.setAdminEmail(request.getAdminEmail());
        college.setPasswordHash(encodedPassword);

        collegeRepository.save(college);

        Admin admin = new Admin();
        admin.setAdminName(request.getAdminName());
        admin.setAdminEmail(request.getAdminEmail());
        admin.setPasswordHash(encodedPassword);
        admin.setCollege(college);

        adminRepository.save(admin);

        return "Admin signup successful";
    }

    public AdminLoginResponse login(AdminLoginRequest request) {

        Optional<Admin> adminOpt = adminRepository
                .findByAdminEmail(request.getAdminEmail());

        if (adminOpt.isEmpty()) {
            return null;
        }

        Admin admin = adminOpt.get();

        if (passwordEncoder.matches(request.getPassword(), admin.getPasswordHash())) {

            AdminLoginResponse response = new AdminLoginResponse();

            response.setAdminName(admin.getAdminName());
            response.setAdminEmail(admin.getAdminEmail());

            College college = admin.getCollege();
            if (college == null) {
                // Auto-heal logic: try to find a college associated with the admin's email
                college = collegeRepository.findByAdminEmail(admin.getAdminEmail()).orElse(null);
                if (college == null) {
                    // Create a new college for this admin
                    college = new College();
                    String generatedCollegeName = admin.getAdminName() != null ? admin.getAdminName() + "'s College" : "My College";
                    college.setCollegeName(generatedCollegeName);
                    college.setAdminEmail(admin.getAdminEmail());
                    college.setPasswordHash(admin.getPasswordHash());
                    college = collegeRepository.save(college);
                }
                admin.setCollege(college);
                adminRepository.save(admin);
            }

            if (college != null) {
                response.setCollegeName(college.getCollegeName());
            }

            return response;

        } else {
            return null;
        }
    }
}