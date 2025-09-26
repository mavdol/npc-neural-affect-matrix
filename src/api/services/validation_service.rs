use std::ffi::CStr;
use std::os::raw::c_char;
use crate::api::types::ApiResult;

pub fn parse_c_string(ptr: *const c_char, field_name: &str) -> Result<String, *mut ApiResult> {
    if ptr.is_null() {
        return Err(Box::into_raw(Box::new(ApiResult::error(format!("{} is null", field_name)))));
    }

    unsafe {
        match CStr::from_ptr(ptr).to_str() {
            Ok(s) => Ok(s.to_string()),
            Err(_) => Err(Box::into_raw(Box::new(ApiResult::error(format!("Invalid UTF-8 string for {}", field_name))))),
        }
    }
}

pub fn parse_optional_c_string(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }

    unsafe {
        CStr::from_ptr(ptr).to_str().ok().map(|s| s.to_string())
    }
}

pub fn validate_config_json(config_str: &str) -> Result<(), *mut ApiResult> {
    if config_str.is_empty() {
        return Err(Box::into_raw(Box::new(ApiResult::error("Config string is empty".to_string()))));
    }
    Ok(())
}

pub fn validate_memory_json(memory_str: &str) -> Result<(), *mut ApiResult> {
    if memory_str.is_empty() {
        return Ok(());
    }
    Ok(())
}
