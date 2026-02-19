<?php

// Get result of validating data

$image_basedir = '/Users/rpage/Library/CloudStorage/GoogleDrive-rdmpage@gmail.com/My Drive/iNat';

$test_filename   = 'val-lep.json';
$result_filename = 'inat2018_test_preds.csv';

// Read data we are testing
$json = file_get_contents($test_filename );

// Read predictions from model
$headings = array($test_filename);
$test = json_decode($json);

// print_r($test);

$images = array();
$categories = array();

$image_to_category = array();

// images
foreach ($test->images as $image)
{
	$images[$image->id] = $image;
}

foreach ($test->categories as $category)
{
	$categories[$category->id] = $category;
	$categories[$category->id]->images = array();
}

foreach ($test->annotations as $annotation)
{
	$categories[$annotation->category_id]->images[$annotation->image_id] = $images[$annotation->image_id];
	
	$image_to_category[$annotation->image_id] = $annotation->category_id;
}


$row_count = 0;

$file_handle = fopen($result_filename, "r");
while (!feof($file_handle)) 
{
	$line = trim(fgets($file_handle));
		
	$row = explode(",",$line);

	$go = is_array($row) && count($row) > 1;
	
	if ($go)
	{
		if ($row_count == 0)
		{
			$headings = $row;		
		}
		else
		{
			$id = $row[0];			
			$predictions = explode(' ', $row[1]);
			
			$cat_id = $image_to_category[$id];
			$categories[$cat_id]->images[$id]->predictions = $predictions;
				
		}
	}
	
	$row_count++;
}


// web page of results

echo '<html>';
echo '<head>
<style>
	/* heavily based on https://css-tricks.com/adaptive-photo-layout-with-flexbox/ */
	.gallery ul {
	  display: flex;
	  flex-wrap: wrap;
	  
	  list-style:none;
	  
	  padding-left:4px;
	}

	.gallery li {
	  height: 100px;
	  flex-grow: 1;
      padding:1em;
	}

	.gallery li:last-child {
	  flex-grow: 10;
	}

	.gallery img {
	  max-height: 90%;
	  min-width: 90%;
	  object-fit: cover;
	  vertical-align: bottom;
	  
	  border:1px solid rgb(192,192,192);
	}	
	
	</style>
</head>';

echo '<body>';

foreach ($categories as $category)
{
	echo '<h2>' . $category->name . ' [' . $category->id . ']</h2>';

	echo '<div class="gallery">';
	echo '<ul>';

	foreach ($category->images as $image)
	{
		echo '<li';
	
		if (in_array($category->id, $image->predictions))
		{
			if ($image->predictions[0] == $category->id)
			{
				echo ' style="background:#2dc937;"';
			}
			else
			{
				echo ' style="background:#e7b416;"';		
			}	
		}
		else
		{
			echo ' style="background:#cc3232;"';
		}
	
		echo '>';
		echo '<img src="' . $image_basedir . '/' . $image->file_name . '" title="[' . $image->id . ']">';	
		echo '</li>';
	}
	echo '<li></li>';
	echo '</ul>';
	echo '</div>';
}

echo '</body>';
echo '</html>';

?>
