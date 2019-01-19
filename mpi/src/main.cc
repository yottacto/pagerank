#include <iostream>
#include "pagerank.hh"
#include "util.hh"
#include "config.hh"

int main()
{
    auto config{ice::config{}};
    auto path = config.front().path;
    auto eps = config.front().eps;

    ice::pagerank pr{path, eps};
    // warm up
    // pr.compute<false>();
    pr.compute<true>();
    pr.print_statistic();
}

