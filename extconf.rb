require 'mkmf'
require 'numo/narray'

opencv4_libs = `pkg-config --libs-only-l opencv4`.chomp
opencv4_cppflags = `pkg-config --cflags-only-I opencv4`.chomp
found_opencv4 = $?.success?

$LOAD_PATH.each{|lp|
    if File.exists?(File.join(lp, 'numo/numo/narray.h'))
    $INCFLAGS = "-I#{lp}/numo #{$INCFLAGS}"
    break
  end
}

if found_opencv4
  $libs = opencv4_libs
  $CPPFLAGS = opencv4_cppflags
  create_makefile('cv2')
end
