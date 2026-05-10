package com.bhuvanshu.mindcare.dto;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class StudentAddRequest {
    private String enrollmentId;
    private String name;
    private Integer age;
    private String gender;
    private String department;
    private String degreeGroup;
}
