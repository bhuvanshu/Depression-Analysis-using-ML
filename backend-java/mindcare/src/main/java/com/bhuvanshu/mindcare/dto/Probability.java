package com.bhuvanshu.mindcare.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class Probability {

    private Double depressed;

    @JsonProperty("not_depressed")
    private Double not_depressed;
}