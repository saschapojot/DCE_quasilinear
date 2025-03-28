#include "./evolution/evolution.hpp"

int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::cout << "wrong arguments" << std::endl;
        std::exit(2);
    }
    auto evo_obj=evolution(std::string(argv[1]));
    evo_obj.init();
    evo_obj.run_and_save_H1R_only();





//     arma::cube C(3,3,3);
//     arma::mat A(3,3);
//     int val=0;
//     for (int k=0;k<3;k++)
//     {
//         for (int i=0;i<3;i++)
//         {
//             for (int j=0;j<3;j++)
//             {
//                 C.slice(k)(i,j)=val++;
//             }
//         }
//     }
//
//
// C.slice(0).print("C.slice(0):");
//
//     C.slice(1).print("C.slice(1):");
//
//     C.slice(2).print("C.slice(2):");
//     val=0;
//     for (int i=0;i<3;i++)
//     {
//         for (int j=0;j<3;j++)
//         {
//             A(i,j)=val++;
//         }
//     }
// A*=2;
//     A.print("A:");
//     arma::cube B(3,3,3);
//     B=A*C.each_slice();
//
//     B.slice(0).print("B.slice(0):");
//
//     B.slice(1).print("B.slice(1):");
//     B.slice(2).print("B.slice(2):");

    return 0;
}
