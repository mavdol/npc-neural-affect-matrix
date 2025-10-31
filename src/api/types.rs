use std::ffi::CString;
use std::os::raw::c_char;

#[repr(C)]
pub struct ApiResult {
    pub success: u8,
    pub data: *mut c_char,
    pub error: *mut c_char,
}

pub type NpcId = String;

impl ApiResult {
    pub fn success(data: String) -> Self {
        let data_ptr = match CString::new(data) {
            Ok(cstring) => cstring.into_raw(),
            Err(_) => CString::new("Data contains invalid characters")
                .expect("Hardcoded fallback string should be valid")
                .into_raw(),
        };

        Self {
            success: 1,
            data: data_ptr,
            error: std::ptr::null_mut(),
        }
    }

    pub fn error(error: String) -> Self {
        let error_ptr = match CString::new(error) {
            Ok(cstring) => cstring.into_raw(),
            Err(_) => CString::new("Error message contains invalid characters")
                .expect("Hardcoded fallback string should be valid")
                .into_raw(),
        };

        Self {
            success: 0,
            data: std::ptr::null_mut(),
            error: error_ptr,
        }
    }
}
