#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "opentelemetry-cpp::prometheus_exporter" for configuration "Release"
set_property(TARGET opentelemetry-cpp::prometheus_exporter APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opentelemetry-cpp::prometheus_exporter PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libopentelemetry_exporter_prometheus.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libopentelemetry_exporter_prometheus.dylib"
  )

list(APPEND _cmake_import_check_targets opentelemetry-cpp::prometheus_exporter )
list(APPEND _cmake_import_check_files_for_opentelemetry-cpp::prometheus_exporter "${_IMPORT_PREFIX}/lib/libopentelemetry_exporter_prometheus.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
