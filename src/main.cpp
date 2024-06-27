#include <iostream>
#include "opencv.hpp"

int main()
{
  try
  {
    std::cout << "World!\n" << std::endl;
    cv::Mat scr = cv::imread("lena.jpg");
  }
  catch (const std::exception& e)
  {
    std::cerr << "Caught exception:\n"
              << e.what() << "\n";
  }
  catch (...)
  {
    std::cerr << "Caught unknown exception\n";
  }

  return EXIT_SUCCESS;
}
