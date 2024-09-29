#include <iostream>

#include "DGtal/io/readers/MeshReader.h"
#include "DGtal/base/Common.h"
#include "DGtal/io/DrawWithDisplay3DModifier.h"
#include "DGtal/io/Color.h"
#include "DGtal/helpers/StdDefs.h"
#include "DGtal/shapes/Shapes.h"
#include "DGtal/io/viewers/Viewer3D.h"
#include "DGtal/shapes/MeshVoxelizer.h"
#include "DGtal/images/RigidTransformation3D.h"
#include "DGtal/kernel/BasicPointFunctors.h"
#include "DGtal/io/writers/GenericWriter.h"
#include <DGtal/math/linalg/EigenDecomposition.h>
#include "DGtal/io/writers/PGMWriter.h"

#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "CLI11.hpp"
#define PI 3.14159265

using namespace DGtal;
using namespace Z3i;
using namespace functors;
using namespace pcl;

//pour le calcul des direction principales
typedef PointVector<9, double> Matrix3x3Point;
typedef SimpleMatrix<double, 3, 3 > CoVarianceMat;
//pour le calcul des normales
typedef typename Mesh<Z3i::RealPoint>::MeshFace Face;
//pour la projection sur les images de profile
typedef ImageContainerBySTLVector<Z3i::Domain, unsigned int> Image3D;
typedef ImageContainerBySTLVector < Z3i::Domain, Z3i::RealPoint > VectorFieldImage3D;
typedef ImageContainerBySTLVector < Z2i::Domain, unsigned char> Image2D;
typedef DGtal::ConstImageAdapter<Image3D, Z2i::Domain, DGtal::functors::Point2DEmbedderIn3D<DGtal::Z3i::Domain>,
                                    Image3D::Value,  DGtal::functors::Identity >  ImageAdapterExtractor;

template<typename TImage, typename TVectorField, typename TPoint>





void
fillPointArea(TImage &anImage, TVectorField &imageVectorField, const TPoint aPoint, const unsigned int size, const unsigned int value, const Z3i::RealPoint &n){

  //typename TImage::Domain aDom(aPoint-TPoint::diagonal(size), aPoint+TPoint::diagonal(size));
  typename TImage::Domain aDom(aPoint - Z3i::Point(1, 1, 1), aPoint + Z3i::Point(0, 0, 0));
  for(typename TImage::Domain::ConstIterator it= aDom.begin(); it != aDom.end(); it++){

    if (anImage.domain().isInside(*it)){
      //anImage.setValue(*it, anImage(*it)+value);
      anImage.setValue(*it, value);
      imageVectorField.setValue(*it, n);
    }

  }
}

// Fonction pour remplir un plan de voxels perpendiculaire à un axe principal dans le domaine spécifié
void fillVoxelPlaneInDomain(Image3D &image, const Z3i::RealPoint &normal, const Z3i::Point &center, unsigned int value) {
    // Calcul de la constante du plan d = n·c
    double d = normal.dot(center);

    // Parcours du domaine de l'image
    for (auto it = image.domain().begin(); it != image.domain().end(); ++it) {
        Z3i::Point pt = *it;
        // Calcul de la distance entre le point et le plan |n·p - d|
        double dist = std::abs(normal[0] * pt[0] + normal[1] * pt[1] + normal[2] * pt[2] - d);

        // Si le point est proche du plan (distance < 0.5), on le marque
        if (dist < 0.5) { 
            image.setValue(pt, value); // Colorer le voxel avec la valeur donnée (par exemple, 255 pour rouge)
        }
    }
}


std::vector<Z3i::RealPoint>
computeNormal(DGtal::Mesh<Z3i::RealPoint> mesh) {
  std::vector<Z3i::RealPoint> vectorOfNormals;
  for (int i = 0; i < mesh.nbFaces(); i++){
    Face currentFace = mesh.getFace(i);
    Z3i::RealPoint currentFaceP0 = mesh.getVertex(currentFace.at(0));
    Z3i::RealPoint currentFaceP1 = mesh.getVertex(currentFace.at(1));
    Z3i::RealPoint currentFaceP2 = mesh.getVertex(currentFace.at(2));
    Z3i::RealPoint center = (currentFaceP0+currentFaceP1+currentFaceP2)/3.0;
    Z3i::RealPoint v0 = currentFaceP0 - currentFaceP2;
    Z3i::RealPoint v1 = currentFaceP1 - currentFaceP2;
    Z3i::RealPoint vnormal=v0.crossProduct(v1);
    Z3i::RealPoint vnormalised(vnormal/vnormal.norm());
    vectorOfNormals.push_back(vnormalised);
  }
  return vectorOfNormals;
}

static CoVarianceMat
getCoVarianceMatFrom(const Matrix3x3Point &aMatrixPt){
  CoVarianceMat res;
  for(unsigned int i = 0; i<3; i++){
    for(unsigned int j = 0; j<3; j++){
      res.setComponent(i, j, aMatrixPt[j+i*3] );
    }
  }
  return res;
}

std::pair<DGtal::Z3i::RealPoint, DGtal::Z3i::RealPoint>
getMainDirsCoVar(Mesh<Z3i::RealPoint>::Iterator begin, Mesh<Z3i::RealPoint>::Iterator end )
{
  std::pair<DGtal::Z3i::RealPoint, DGtal::Z3i::RealPoint> res;
  Matrix3x3Point c ;
  unsigned int nb = 0;
  Z3i::RealPoint centroid;
  int cp=0;
  for(auto it=begin; it != end; it++){
      Z3i::RealPoint pt=*it;
      centroid[0] += pt[0];
      centroid[1] += pt[1];
      centroid[2] += pt[2];
       ++cp;
  }
  centroid /=cp;
  for(auto it=begin; it != end; it++){
    
      Z3i::RealPoint pt=*it;
      pt[0] -= centroid[0];
      pt[1] -= centroid[1];
      pt[2] -= centroid[2];

      c[4] += pt[1] * pt[1];
      c[5] += pt[1] * pt[2];
      c[8] += pt[2] * pt[2];
      pt *= pt[0];
      c[0] += pt[0];
      c[1] += pt[1];
      c[2] += pt[2];
      nb++;
  }
  c[3] = c[1];
  c[6] = c[2];
  c[7] = c[5];
  c = c/nb;
  CoVarianceMat covar = getCoVarianceMatFrom(c);

  SimpleMatrix<double, 3, 3 > eVects;
  PointVector<3, double> eVals;
  DGtal::EigenDecomposition<3, double, CoVarianceMat>::getEigenDecomposition (covar, eVects, eVals);
  unsigned int temp = 0;
  unsigned int major_index = 0;
  unsigned int middle_index = 1;
  unsigned int minor_index = 2;
  
  res.first = eVects.column(major_index);
  res.second =  eVects.column(middle_index);

  return res;
}

Image2D 
compute2D_profile(Z3i::RealPoint mainDir, Z3i::RealPoint secDir, double widthImageScan,
                  Image3D meshVolImage, Image2D::Domain aDomain2D, int maxScan){
    double nx = mainDir[0];
    double ny = mainDir[1];
    double nz = mainDir[2];
    Z3i::Point ptL = meshVolImage.domain().lowerBound();
    Z3i::Point ptU = meshVolImage.domain().upperBound();
    Z3i::Point ptC = (ptL+ptU)/2;
    int centerX=ptC[0]-(double)maxScan*nx/2;
    int centerY=ptC[1]-(double)maxScan*ny/2;
    int centerZ=ptC[2]-(double)maxScan*nz/2;
    Z3i::Point ptCenter (centerX, centerY, centerZ);
    Z3i::Point c (ptCenter-mainDir, DGtal::functors::Round<>());
    //domaine de l'image de profile
    Image2D resultingImage(aDomain2D);
    //init all value to 0
    for(Image2D::Domain::ConstIterator it = resultingImage.domain().begin();it != resultingImage.domain().end(); it++){
        resultingImage.setValue(*it, 0);
    }
    //embedder 3D to 2D (loop over 3d volume by image loop)
    DGtal::functors::Point2DEmbedderIn3D<DGtal::Z3i::Domain >  embedder(meshVolImage.domain(),
                                                                        c,
                                                                        mainDir,
                                                                        secDir,
                                                                        widthImageScan);
    //identity (but we can make rotate or translate here)
    DGtal::functors::Identity idV;
    //extractor 3d to 2D
    ImageAdapterExtractor extractedImage(meshVolImage, aDomain2D, embedder, idV);
    //limit loop
    int k = 0;
    //start of shape in 3D
    bool firstFound = false;
    while (k < maxScan || !firstFound){
      trace.progressBar(k, maxScan);
      embedder.shiftOriginPoint(mainDir);
      for(Image2D::Domain::ConstIterator it = extractedImage.domain().begin();it != extractedImage.domain().end(); it++){
        if(resultingImage(*it)== 0 &&  extractedImage(*it)!=0)
          {
            if (!firstFound) {
              firstFound = true;
            }
            resultingImage.setValue(*it, 255);
          }
        }
        if (firstFound){
          k++;
        }
    }
    trace.info()<< std::endl;     
    //output resulting image
    return resultingImage;
}

// Fonction pour déplacer un point `center` d'un décalage le long d'une direction donnée
Z3i::Point 
moveCenterAlongDirection(const Z3i::Point &center, const Z3i::RealPoint &direction, int offset) {
    // Calcul du nouveau centre en déplaçant le centre le long de la direction
    Z3i::RealPoint normalizedDirection = direction.getNormalized(); // Normalisation de la direction
    Z3i::RealPoint offsetVector = normalizedDirection * offset; // Vecteur de déplacement
    Z3i::Point newCenter = center + Z3i::Point(round(offsetVector[0]), round(offsetVector[1]), round(offsetVector[2]));
    return newCenter;
}

/**************************/
/*Fonction pour affichage.*/
/**************************/
// Fonction pour ajouter un plan de voxel perpendiculaire a une normal au viewer.
void 
displayPlane(Viewer3D<> &viewer, Image3D::Domain PCLDo, Color c, Z3i::Point offsetCenter, DGtal::Z3i::RealPoint n) {
  //domaine du plan est le meme que celui du volume 3D
  Image3D voxelPlaneImage(PCLDo);
  for (Image3D::Domain::ConstIterator it = voxelPlaneImage.domain().begin(); it != voxelPlaneImage.domain().end(); it++) {
    voxelPlaneImage.setValue(*it, 0); // Initialiser à zéro
  }
  // Remplir le plan de voxels perpendiculaire à la direction principale avec la couleur rouge
  fillVoxelPlaneInDomain(voxelPlaneImage, n, offsetCenter, 255); // 255 pour le rouge
  // Afficher voxelPlaneImage
  viewer << CustomColors3D(c, c); // Couleur rouge pour le plan
  for (auto it = PCLDo.begin(); it != PCLDo.end(); ++it) {
    Z3i::Point pt = *it;
    if (voxelPlaneImage(pt) > 0) { // Si le voxel est sur le plan, l'afficher
      viewer << pt; 
    }
  }
}

// Fonction pour afficher les directions principales au viewer.
void 
displayMainAxis(Viewer3D<> &viewer, double aLen, DGtal::Z3i::RealPoint n1,DGtal::Z3i::RealPoint n2, Z3i::Point c) {
  // Draw main direction (principal axis)
  viewer << CustomColors3D(Color::Red, Color::Red); // Red for main axis
  viewer.addLine(c, c + n1.getNormalized() * aLen);
  // Draw secondary direction (secondary axis)
  viewer << CustomColors3D(Color::Green, Color::Green); // Green for secondary axis
  viewer.addLine(c, c + n2.getNormalized() * aLen);
  // Draw third axis (perpendicular to mainDir and secDir)
  Z3i::RealPoint n3 = n1.crossProduct(n2);
  viewer << CustomColors3D(Color::Blue, Color::Blue); // Blue for third axis
  viewer.addLine(c, c + n3 * aLen);
}

int
main(int argc,char **argv)
{
    std::string inputFileName;
    std::string outputPrefix="output";
    double imageSize = {200};
    double minDiagDist = std::sqrt((imageSize*imageSize)+
                                                    (imageSize*imageSize));
    int maxScan {80};

    CLI::App app;
    app.description("Allowed options are: ");
    app.add_option("-i,--input,1", inputFileName , "input mesh.")
      ->required()
      ->check(CLI::ExistingFile);
    app.add_option("--output,-o,2", outputPrefix, "output prefix."); 
    app.get_formatter()->column_width(40);
    CLI11_PARSE(app, argc, argv);

    /**prepare mesh**/
    DGtal::Mesh<Z3i::RealPoint> oriMesh(true);
    MeshReader<Z3i::RealPoint>::importOFFFile(inputFileName, oriMesh, false);
    Mesh<Z3i::RealPoint> transMesh = DGtal::Mesh<Z3i::RealPoint>(oriMesh);
    int meshScale=-1;
    std::pair<Z3i::RealPoint, Z3i::RealPoint> b = oriMesh.getBoundingBox();
    double diagDist = (b.first-b.second).norm();
    
    trace.info() << "modify mesh scale... (to "<< imageSize<<")";
    meshScale = imageSize/diagDist;
    oriMesh.rescale(meshScale);
    trace.info() << " \t[done] "<< std::endl ;

    int triangleAreaUnit = 1.;
    triangleAreaUnit *= meshScale;
    oriMesh.vertexBegin();
    oriMesh.quadToTriangularFaces();
    int maxArea = triangleAreaUnit+1.0 ;
    while (maxArea> triangleAreaUnit)
    {
      trace.info()<< "Iterating mesh subdivision... ";
      maxArea = oriMesh.subDivideTriangularFaces(triangleAreaUnit);
      trace.info() << " \t[done]"<< std::endl;
    }
    for (int i = 0; i < oriMesh.nbFaces(); i++)
    {
        // Assigner une couleur en fonction de l'index de la face (par exemple, une couleur changeante)
        Color faceColor = Color((i * 10) % 256, (i * 20) % 256, (i * 30) % 256, 100); // Couleurs semi-transparentes
        oriMesh.setFaceColor(i,faceColor);
    }
    /**compute Main dir**/
    auto dir = getMainDirsCoVar(oriMesh.vertexBegin(), oriMesh.vertexEnd());  
    /**compute normales**/
    trace.info()<< "Starting compute normals... ";
    std::vector<Z3i::RealPoint> normals = computeNormal(oriMesh);
    trace.info() << " \t[done]"<< std::endl;
    /**fill a 3d volume**/
    Image3D::Domain PCLDomain( oriMesh.getBoundingBox().first-Z3i::Point::diagonal(1),
                                oriMesh.getBoundingBox().second+Z3i::Point::diagonal(1));      
      
                         
    Image3D meshVolImage(PCLDomain);
    VectorFieldImage3D meshNormalImage(PCLDomain);
    for(Image3D::Domain::ConstIterator it = meshVolImage.domain().begin(); it != meshVolImage.domain().end(); it++)
    {
      meshVolImage.setValue(*it, 0);
      meshNormalImage.setValue(*it, Z3i::RealPoint(0.0,0.0,0.0));
    }
    trace.info()<< "Fill 3d vol with "<< oriMesh.nbFaces() <<" faces...";
    
    for (int i = 0; i < oriMesh.nbFaces(); i++){
      Face currentFace = oriMesh.getFace(i);
      Z3i::RealPoint aPoint = (oriMesh.getVertex(currentFace.at(0))+
                                oriMesh.getVertex(currentFace.at(1))+
                                  oriMesh.getVertex(currentFace.at(2)))/3.0;
      Z3i::RealPoint aNormals  = normals[i];
      fillPointArea(meshVolImage, meshNormalImage, aPoint, 1, 1, aNormals);
    }
    trace.info() << " \t[done]"<< std::endl;

    /**create profile image**/
    Image2D::Domain aDomain2D(DGtal::Z2i::Point(0,0),DGtal::Z2i::Point(imageSize-1, imageSize-1));
    trace.info()<< "compute first profile ... "<< std::endl;
    Image2D profil_main = compute2D_profile(dir.first.getNormalized(),
                                      dir.second.getNormalized(), 
                                      imageSize,
                                      meshVolImage, 
                                      aDomain2D, 
                                      maxScan);
    
    trace.info()<< "compute second profile ... "<< std::endl;
    Image2D profil_second = compute2D_profile(dir.second.getNormalized(),
                                      dir.first.getNormalized(), 
                                      imageSize,
                                      meshVolImage, 
                                      aDomain2D, 
                                      maxScan);
    PGMWriter<Image2D>::exportPGM(""+outputPrefix+"_m.pgm", profil_main);
    PGMWriter<Image2D>::exportPGM(""+outputPrefix+"_s.pgm", profil_second);

    /**viewer**/
    QApplication application(argc,argv);
    Viewer3D<> viewer;
    viewer.show();
    
    /***Ajoutez l'image 3D au viewer ***/ 
    Z3i::Point center = (PCLDomain.lowerBound() + PCLDomain.upperBound()) / 2; // Centre du domaine
    
    viewer << CustomColors3D(Color::Blue, Color::Blue); // Red for main axis
    for (auto it = PCLDomain.begin(); it != PCLDomain.end(); ++it) {
        Z3i::Point pt = *it;
        if (meshVolImage(pt) > 0) {
            viewer << pt; // Ajouter le voxel transformé à la vue
        }
    }
    /* Affichage du plan principale */
    Z3i::Point offsetCenterPrin = moveCenterAlongDirection(center, dir.first.getNormalized(), 0);
    displayPlane(viewer, PCLDomain, Color(255, 0, 0, 100), offsetCenterPrin, dir.first.getNormalized());
    /* Affichage du plan secondaire */
    Z3i::Point offsetCenterSec = moveCenterAlongDirection(center, dir.second.getNormalized(), 0);
    displayPlane(viewer, PCLDomain, Color(0, 255, 0, 100), offsetCenterSec, dir.second.getNormalized()); 
    /* Affichage des axes principaux */
    double axisLength = imageSize / 2;
    displayMainAxis(viewer, axisLength, dir.first.getNormalized(),dir.second.getNormalized(), center);
    
    
    viewer << Viewer3D<>::updateDisplay;
    return application.exec();
}
/***Ajoutez les axes principaux***/ 
/*
   // Définir des couleurs personnalisées pour les voxels (ici, des voxels bleus avec une transparence)
    viewer << oriMesh;
    Z3i::RealPoint center = (oriMesh.getBoundingBox().first + oriMesh.getBoundingBox().second) / 2.0;
    double axisLength = imageSize / 2;
    // Draw main direction (principal axis)
    viewer << CustomColors3D(Color::Red, Color::Red); // Red for main axis
    viewer.addLine(center, center + dir.first * axisLength);
    // Draw secondary direction (secondary axis)
    viewer << CustomColors3D(Color::Green, Color::Green); // Green for secondary axis
    viewer.addLine(center, center + dir.second * axisLength);
    // Draw third axis (perpendicular to mainDir and secDir)
    Z3i::RealPoint thirdDir = dir.first.crossProduct(dir.second).getNormalized();
    viewer << CustomColors3D(Color::Blue, Color::Blue); // Blue for third axis
    viewer.addLine(center, center + thirdDir * axisLength);
*/

    