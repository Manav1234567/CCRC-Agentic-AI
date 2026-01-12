#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "ArrowDataset::arrow_dataset_shared" for configuration "Release"
set_property(TARGET ArrowDataset::arrow_dataset_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ArrowDataset::arrow_dataset_shared PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "opentelemetry-cpp::trace;opentelemetry-cpp::logs;opentelemetry-cpp::otlp_http_log_record_exporter;opentelemetry-cpp::ostream_log_record_exporter;opentelemetry-cpp::ostream_span_exporter;opentelemetry-cpp::otlp_http_exporter"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libarrow_dataset.2200.0.0.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libarrow_dataset.2200.dylib"
  )

list(APPEND _cmake_import_check_targets ArrowDataset::arrow_dataset_shared )
list(APPEND _cmake_import_check_files_for_ArrowDataset::arrow_dataset_shared "${_IMPORT_PREFIX}/lib/libarrow_dataset.2200.0.0.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
