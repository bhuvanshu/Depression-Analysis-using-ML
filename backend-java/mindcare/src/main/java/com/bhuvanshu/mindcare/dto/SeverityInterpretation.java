package com.bhuvanshu.mindcare.dto;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class SeverityInterpretation {
    private String level;
    private Double score;
    private String meaning;
    private String color;
}
